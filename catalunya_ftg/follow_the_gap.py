#!/usr/bin/env python3
"""
Follow-the-Gap reactive controller for the F1TENTH gym (ROS 2 Humble), v3.
Catalunya F1 track. LiDAR-only driving.

Pipeline per LiDAR scan
  Follow-the-Gap (disparity extension + safety bubble + deepest-gap centre)
      -> pure-pursuit steering proposal
      -> swept-path safety layer: 41 candidate arcs, each scored by clearance AND by the
         speed it would allow (reactive "fast line")
      -> speed = min(distance profile, braking limit, lateral-acceleration limit)
      -> recovery state machine (contact / stuck / blocked) with a rotation cap

Fair play
  * Driving decisions use ONLY the LiDAR scan + wheel speed. No map, no localisation.
    (use_odom_heading:=false is the default; the odometry pose feeds only the lap
    counter, lap timer and HUD, i.e. scoring.)
  * The centerline CSV is read only to DRAW the road surface in RViz.

Learning (mode:=learn, default)
  Learning runs continuously until Ctrl-C. Every lap/collision updates learned_params.json;
  mode:=race loads the best clean aggression value, applies race_margin, and keeps it fixed.

Start pose
  The car is teleported to the Catalunya start pose (x=0, y=0, yaw=-2.14363 rad, i.e. along
  the track) through /initialpose, and the checkered start/finish line is drawn there.

Run
  source /opt/ros/humble/setup.bash
  source ~/F1Tenth-Repository/install/setup.bash
  python3 follow_the_gap.py                                # learn mode
  python3 follow_the_gap.py --ros-args -p mode:=race       # learned value minus a safety margin
  python3 follow_the_gap.py --ros-args -p mode:=fixed -p aggr:=1.2
  # RViz: rviz2 -d catalunya.rviz   (use decimals for float params!)
"""

import collections
import csv
import json
import math
import os
import sys
import time

import numpy as np

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Point, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker
    ROS_OK = True
except ImportError:          # lets the driver logic be tested without ROS
    ROS_OK = False
    Node = object

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hud import render as hud_render, HUD_OK
except ImportError:
    HUD_OK = False


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Tunable parameters (all can be overridden with --ros-args -p name:=value)
# ---------------------------------------------------------------------------
PARAMS = {
    # car geometry (F1TENTH gym defaults); LiDAR sits at the car reference point
    'wheelbase': 0.3302,    # F1TENTH Gym lf + lr
    'lr': 0.17145,          # reference point -> rear axle
    'half_len': 0.29,       # body half length (gym length 0.58)
    'half_w': 0.155,        # body half width  (gym width 0.31)
    'safety_margin': 0.10, # body clearance for the swept path and disparity widening (m)
    # perception / Follow-the-Gap
    'fov_deg': 90.0,
    'max_range': 12.0,      # controller planning horizon from the requested LiDAR range
    'smooth_window': 5,
    'disparity_thresh': 0.5,
    'gap_tolerance': 0.25,  # distance tolerance for choosing the deepest part of a gap (m)
    'bubble_radius': 0.35,
    'free_threshold': 0.8,
    'gap_depth_clip': 5.0,
    # control
    'max_steer': 0.42,      # requested steering limit; Gym dynamics clip to their physical limit
    'steer_gain': 0.9,
    'steer_smooth': 0.7,    # alpha on the newly selected steering command
    'lookahead_min': 0.7,
    'lookahead_max': 1.5,
    'n_candidates': 41,     # keep ODD so that 0.0 (straight) is a candidate
    'd_look': 12.0,         # do not plan beyond the configured LiDAR horizon
    'w_ftg': 2.0,           # prioritize the chosen free-space direction
    'w_center': 0.3,        # keep lane centring without overpowering the steering target
    'center_s': 2.5,        # how far along each arc (m) the centring is judged
    'center_win': 1.0,      # +-m along the arc heading used to find left/right wall distance
    'outer_bias': 0.0,      # center the lane target instead of deliberately taking a wide entry
    'w_speed': 0.3,         # preference for the arc that allows the highest speed (fast line)
    'aggr': 1.0,            # aggression knob (learned): scales target speed AND lateral-accel budget
    'use_odom_heading': False,  # False = LiDAR-only driving (fair mode)
    'w_clear': 0.4,         # clearance is also enforced as a hard stopping constraint
    'soft_margin': 0.25,    # extra "comfort" margin used only for scoring (m)
    'w_soft': 0.7,          # comfort margin after the hard clearance constraint
    'a_dec': 6.0,           # requested braking deceleration (m/s^2)
    'front_decel': 6.0,     # braking deceleration for an obstacle in the narrow front cone
    'front_half_angle_deg': 6.0,
    'a_lat': 7.0,           # requested lateral acceleration (m/s^2)
    'a_lat_cap': 7.0,       # learning cannot raise the requested lateral limit
    'v_max': 12.0,          # requested maximum speed (m/s)
    'req_speed_cap': 11.6,  # keep stopping distance within the 12 m LiDAR planning horizon
    'w_cont': 0.5,          # retain continuity without resisting tighter turns
    'curv_alpha': 0.5,      # low-pass weight on newly observed target curvature
    'stop_dist': 0.35,      # clearance kept beyond the predicted stopping point (m)
    'min_speed': 0.0,       # allow the brake limiter to command a complete stop
    'd_block': 0.30,        # every steering option shorter than this -> back out
    'speed_scale': 1.0,     # master speed knob
    'speed_dist': [0.5, 1.5, 3.0, 6.0, 10.0, 12.0],
    'speed_val': [1.5, 3.0, 6.0, 9.0, 11.0, 12.0],
    # direction awareness / recovery
    'wrong_way_rad': 1.9,   # nose vs track heading error that means "wrong way"
    'stuck_time': 1.2,      # s of commanded motion without moving
    'rev_speed': 0.35,      # MUST stay < 0.5 m/s: the gym's dynamic tyre model goes unstable when reversing faster
    'rev_time': 1.6,
    'turn_speed': 0.9,
    'rear_clear': 0.38,     # stop reversing if the rear is this close (m)
    'recovery_timeout': 12.0,
    'brake_time': 0.8,      # max time to brake to a stop before reversing (s)
    'blocked_time': 0.25,   # 'no safe arc' must persist this long (and car be slow) to trigger recovery
    'recovery': False,      # physical recovery manoeuvres OFF: any crash = counted collision + teleport to the line
    'spin_margin': 1.0,     # rad/s above the kinematic yaw rate that means 'spinning out'
    # trouble-spot memory (learns WHERE the car struggles; uses wheel-speed path length only)
    'memory': True,
    'mem_bucket': 5.0,      # m of path length per memory cell
    'mem_back': 6,          # cells before the trouble spot that get adjusted (6 x 5 m = 30 m)
    'mem_lat_step': 0.25,   # how much each event shifts the preferred lane (0..1)
    'mem_cap_min': 0.6,     # a crash never lowers the local speed scale below this
    'mem_crawl_speed': 1.5, # m/s: slower than this for mem_crawl_time = 'crawl' event
    'mem_crawl_time': 0.3,
    'rot_limit': 1.2,       # max net rotation (rad) allowed during one recovery: keeps the car facing forward
}


# ===========================================================================
# ROS-independent driving logic
# ===========================================================================
class FTGDriver:
    """LiDAR + pose in, (speed, steering) out. No ROS needed."""

    def __init__(self, P, ref_heading=None):
        self.P = dict(P)
        self.ref = ref_heading          # learned track heading (rad)
        self.yaw = None
        self.v = 0.0
        self.last_pos = None
        self.last_xy = None             # for visuals only
        self.wrong_way = False
        self._ww_since = None

        self.prev_steer = 0.0
        self.cmd_speed = 0.0
        self.curvature = 0.0
        self._stuck_since = None

        self.mode = 'DRIVE'             # DRIVE | REVERSE | TURN
        self.mode_t0 = 0.0
        self.rec_t0 = 0.0
        self.turn_dir = 1.0
        self.rev_steer = 0.0
        self.recoveries = 0
        self.last_info = {}
        self.psi_dr = 0.0               # dead-reckoned heading (wheel speed + steering), LiDAR-only mode
        self.t_dr = None
        self.last_out_steer = 0.0
        self.hist = collections.deque(maxlen=80)   # (t, psi_dr, gap angle): where 'forward' was
        self.fwd_est = None
        self._blocked_t = None
        self.psi_rec = 0.0              # dead-reckoned rotation since recovery start
        self.t_prev = 0.0
        self.last_steer_cmd = 0.0
        self.yaw_rate = 0.0
        self._spin_since = None
        self.crash_reason = ''
        self.crashed = False
        self.s = 0.0                    # path length since the line (wheel speed integrated)
        self.mem_n = 400
        self.mem_lat = np.zeros(self.mem_n)     # preferred lane offset (+ = car right of centre for a left turn)
        self.mem_cap = np.ones(self.mem_n)      # local speed scale
        self.mem_hit = np.zeros(self.mem_n, dtype=bool)
        self._crawl_t = None
        self._crawl_cell = 0
        self._crawl_fired = False

        self.in_contact = False
        self.collision_count = 0
        self.blocked = False

        self._bc = None
        self._px = np.zeros(0)
        self._py = np.zeros(0)

    # ------------------------------------------------------------------
    # 1. Sense of direction
    # ------------------------------------------------------------------
    def update_pose(self, x, y, yaw, v, t, yaw_rate=0.0):
        """Feed odometry. Learns the track heading and detects 'wrong way'."""
        self.yaw, self.v = yaw, v
        self.yaw_rate = yaw_rate
        if not self.P['use_odom_heading']:
            return                      # LiDAR-only mode: pose is NOT used for driving
        if self.ref is None:
            self.ref = yaw
        if self.last_pos is None:
            self.last_pos = (x, y)
            return

        dx, dy = x - self.last_pos[0], y - self.last_pos[1]
        if math.hypot(dx, dy) >= 0.3:
            trav = math.atan2(dy, dx)
            # Only learn from motion that is forward (nose along the movement)
            # and consistent with the current reference (not a reversed drive).
            if abs(wrap(trav - self.ref)) < 1.2 and abs(wrap(trav - yaw)) < 1.2:
                self.ref = wrap(self.ref + 0.5 * wrap(trav - self.ref))
            self.last_pos = (x, y)

        err = abs(wrap(yaw - self.ref))
        if err > self.P['wrong_way_rad']:
            if self._ww_since is None:
                self._ww_since = t
            if t - self._ww_since > 0.3:
                self.wrong_way = True
        else:
            self._ww_since = None
            if err < 0.9:
                self.wrong_way = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _beams(self, n, angle_min, inc):
        key = (n, round(angle_min, 5), round(inc, 7))
        if self._bc is None or self._bc[0] != key:
            ang = angle_min + inc * np.arange(n)
            c, s = np.cos(ang), np.sin(ang)
            dcar = np.minimum(self.P['half_len'] / np.maximum(np.abs(c), 1e-9),
                              self.P['half_w'] / np.maximum(np.abs(s), 1e-9))
            self._bc = (key, ang, dcar, c, s)
        return self._bc[1:]

    def _arc_clear(self, delta, extra=0.0):
        """Free distance (m) the car can drive with constant steering `delta`
        before its body (with margins) would hit a LiDAR point.
        `extra` widens the corridor (used for the soft comfort margin)."""
        P = self.P
        d_look = P['d_look']
        px, py = self._px, self._py
        if px.size == 0:
            return d_look
        w = P['half_w'] + P['safety_margin'] + extra
        front = P['half_len'] + P['safety_margin']

        if abs(delta) < 1e-3:                       # straight line
            m = (np.abs(py) < w) & (px > 0.0)
            if not np.any(m):
                return d_look
            return float(min(max(np.min(px[m]) - front, 0.0), d_look))

        lr = P['lr']
        R = P['wheelbase'] / math.tan(delta)        # signed radius (rear axle)
        s = 1.0 if R > 0 else -1.0
        aR = abs(R)
        xr = px + lr                                # relative to turn center
        yr = py - R
        rho = np.hypot(xr, yr)
        xf = front + lr                             # front extent from axle
        r_in = aR - w
        r_out = math.hypot(aR + w, xf)
        m = (rho >= r_in) & (rho <= r_out)
        if not np.any(m):
            return d_look
        xm, ym, rm = xr[m], yr[m], rho[m]
        theta = np.arctan2(xm, -s * ym)             # angle ahead of axle line
        rho_f = math.hypot(xf, r_in)
        lead = np.where(
            rm >= rho_f,
            np.arcsin(np.minimum(xf / rm, 1.0)),                       # front edge
            np.arcsin(np.sqrt(np.maximum(rm * rm - r_in * r_in, 0.0)) / rm))  # side
        ahead = theta - lead
        ahead = ahead[ahead >= -0.02]
        if ahead.size == 0:
            return d_look
        return float(min(max(np.min(ahead), 0.0) * aR, d_look))

    def _lane_balance(self, deltas, dist):
        """For each candidate arc: (gl - gr)/(gl + gr) at the arc point `center_s` ahead
        (+ = more room on the left). Uses only the LiDAR points."""
        P = self.P
        px, py = self._px, self._py
        out = np.zeros(len(deltas))
        if px.size == 0:
            return out
        L, lr, win, cap = P['wheelbase'], P['lr'], P['center_win'], 3.0
        for k, d in enumerate(deltas):
            s = min(P['center_s'], max(float(dist[k]) - 0.4, 0.4))
            if abs(d) < 1e-3:
                ex, ey, ph = s, 0.0, 0.0
            else:
                R = L / math.tan(d)
                ph = s / R
                ex = -lr + R * math.sin(ph) + lr * math.cos(ph)
                ey = R * (1.0 - math.cos(ph)) + lr * math.sin(ph)
            c, sn = math.cos(ph), math.sin(ph)
            rx, ry = px - ex, py - ey
            along = rx * c + ry * sn
            lat = -rx * sn + ry * c
            m = np.abs(along) < win
            if not np.any(m):
                continue
            l, r_ = lat[m], -lat[m]
            gl = float(np.min(l[l > 0])) if np.any(l > 0) else cap
            gr = float(np.min(r_[r_ > 0])) if np.any(r_ > 0) else cap
            gl, gr = min(gl, cap), min(gr, cap)
            out[k] = (gl - gr) / max(gl + gr, 1e-3)
        return out

    def _front_clearance(self, raw, ang):
        """Clearance along a narrow forward cone, measured from the car body."""
        P = self.P
        front = np.abs(ang) <= math.radians(P['front_half_angle_deg'])
        if not np.any(front):
            return P['max_range']
        ranges = raw[front]
        ranges = ranges[np.isfinite(ranges) & (ranges > 0.02)]
        if ranges.size == 0:
            return P['max_range']
        body_front = P['half_len'] + P['safety_margin']
        return float(np.clip(np.min(ranges) - body_front, 0.0, P['max_range']))

    def _extend_disparities(self, r, inc):
        out = r.copy()
        wd = self.P['half_w'] + self.P['safety_margin']
        for i in np.where(np.abs(np.diff(r)) > self.P['disparity_thresh'])[0]:
            if r[i] < r[i + 1]:
                closer, start, step = r[i], i + 1, 1
            else:
                closer, start, step = r[i + 1], i, -1
            if closer < 0.05:
                continue
            n = int(math.ceil(math.atan2(wd, closer) / inc))
            for j in range(n):
                k = start + step * j
                if 0 <= k < len(out):
                    out[k] = min(out[k], closer)
        return out

    def _ftg_target(self, raw, ang, inc):
        """Follow-the-Gap: returns (angle, distance) of the point to aim at."""
        P = self.P
        idx = np.where(np.abs(ang) <= math.radians(P['fov_deg']))[0]
        if idx.size == 0:
            idx = np.arange(len(ang))
        lo, hi = idx[0], idx[-1] + 1
        r = raw[lo:hi].copy()
        a = ang[lo:hi]

        k = max(1, int(P['smooth_window']))
        if k > 1:
            pad = np.pad(r, (k // 2, k // 2), mode='edge')
            r = np.convolve(pad, np.ones(k) / k, mode='valid')[:len(r)]

        free = self._extend_disparities(r, inc)              # disparity extender
        c = int(np.argmin(free))                             # safety bubble
        nb = int(math.ceil(math.atan2(P['bubble_radius'], max(free[c], 0.1)) / inc))
        free[max(0, c - nb):c + nb + 1] = 0.0

        mask = (free > P['free_threshold']).astype(int)      # longest gap
        edges = np.diff(np.concatenate(([0], mask, [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        if len(starts) == 0:
            best = int(np.argmax(r))
        else:
            kk = int(np.argmax(ends - starts))
            s, e = starts[kk], ends[kk]
            g = np.minimum(free[s:e], P['gap_depth_clip'])   # center of deepest part
            cands = np.where(g >= g.max() - P['gap_tolerance'])[0]
            best = s + int(np.median(cands))
        return float(a[best]), float(r[best])

    # ------------------------------------------------------------------
    # 3. Normal driving: FTG proposal + swept-path safety layer
    # ------------------------------------------------------------------
    def _plan(self, raw, ang, inc):
        P = self.P
        L, ms = P['wheelbase'], P['max_steer']

        theta, d_t = self._ftg_target(raw, ang, inc)
        ld = min(max(d_t, P['lookahead_min']), P['lookahead_max'])
        d_ftg = P['steer_gain'] * math.atan2(2.0 * L * math.sin(theta), ld)
        d_ftg = max(-ms, min(ms, d_ftg))
        deltas = np.linspace(-ms, ms, int(P['n_candidates']))
        dist = np.array([self._arc_clear(d) for d in deltas])
        dist_soft = np.array([self._arc_clear(d, P['soft_margin']) for d in deltas])

        v_ref = max(abs(self.v), 1.0)
        v_req = min(v_ref, P['req_speed_cap'])
        d_req = P['stop_dist'] + v_req * v_req / (2.0 * P['a_dec'])
        cell = self._cell(self.s + max(self.v, 0.0) * 0.3)
        mem_lat = float(self.mem_lat[cell]) if P['memory'] else 0.0
        aggr = P['aggr'] * (float(self.mem_cap[cell]) if P['memory'] else 1.0)
        a_lat = min(P['a_lat'] * aggr, P['a_lat_cap'])
        v_top = min(P['v_max'], P['speed_val'][-1] * P['speed_scale'] * aggr)
        v_prof_i = np.interp(dist, P['speed_dist'], P['speed_val']) * P['speed_scale'] * aggr
        v_gov_i = np.sqrt(2.0 * P['a_dec'] * np.maximum(dist - P['stop_dist'], 0.0))
        v_lat_i = np.sqrt(a_lat * L / np.maximum(np.abs(np.tan(deltas)), 1e-3))
        v_i = np.minimum(np.minimum(np.minimum(v_prof_i, v_gov_i), v_lat_i), P['v_max'])
        score = (P['w_ftg'] * np.abs(deltas - d_ftg) / ms
                 + P['w_speed'] * (1.0 - np.minimum(v_i / v_top, 1.0))
                 + P['w_cont'] * np.abs(deltas - self.prev_steer) / ms
                 + P['w_clear'] * (1.0 - np.minimum(dist, P['d_look']) / P['d_look'])
                 + P['w_soft'] * (1.0 - np.minimum(dist_soft, P['d_look']) / P['d_look']))
        if P['w_center'] > 0.0:
            q = self._lane_balance(deltas, dist)
            q_star = P['outer_bias'] * min(abs(d_ftg) / ms, 1.0) * (1.0 if d_ftg > 0 else -1.0)
            q_star = max(-1.0, min(1.0, q_star + mem_lat))
            score = score + P['w_center'] * np.abs(q - q_star)
        ok = dist >= d_req
        i = int(np.argmin(np.where(ok, score, np.inf))) if np.any(ok) \
            else int(np.argmax(dist))
        steer_cmd, d_i = float(deltas[i]), float(dist[i])
        d_straight = float(dist[len(dist) // 2])
        self.blocked = float(np.max(dist)) < P['d_block']

        a = P['steer_smooth']
        # Keep steering responsive up to the physical limit. The lateral speed
        # governor below slows the car for the selected curvature; clipping steering
        # by current speed here caused wide arcs while the car was still braking.
        steer = float(np.clip(a * steer_cmd + (1.0 - a) * self.prev_steer, -ms, ms))
        self.prev_steer = steer
        # Filter the curvature of the path actually selected. This keeps small
        # scan-to-scan steering changes from hammering the speed target while the
        # swept-path and stopping-distance checks remain hard constraints.
        curve_now = max(abs(math.tan(steer)), abs(math.tan(steer_cmd))) / L
        curv_alpha = float(np.clip(P['curv_alpha'], 0.0, 1.0))
        self.curvature = curv_alpha * curve_now + (1.0 - curv_alpha) * self.curvature

        v_prof = float(np.interp(d_i, P['speed_dist'], P['speed_val'])) * P['speed_scale'] * aggr
        v_gov = math.sqrt(2.0 * P['a_dec'] * max(d_i - P['stop_dist'], 0.0))
        # corner anticipation: slow down for the turn the gap target asks for,
        # and for a wall approaching straight ahead, before it is too late
        # Keep braking tied to the selected path curvature even when the high-speed
        # steering guard temporarily clips the applied steering angle.
        v_lat = math.sqrt(a_lat / max(self.curvature, 1e-3))
        # wall ahead: judged along the arc the car is ACTUALLY following (applied steering),
        # not along the exactly-straight line, which can clip a wall corner and wrongly crawl
        d_cur = d_i if abs(steer - steer_cmd) < 0.08 else min(d_i, self._arc_clear(steer))
        v_str = math.sqrt(2.0 * P['a_dec'] * max(d_cur - P['stop_dist'], 0.0)) + 0.5
        d_front = self._front_clearance(raw, ang)
        v_front = math.sqrt(P['front_decel'] * 2.0 * max(d_front - P['stop_dist'], 0.0))
        lims = {'profile': v_prof, 'brake': v_gov, 'lateral': v_lat,
                'straight_wall': v_str, 'front_obstacle': v_front}
        why = min(lims, key=lims.get)
        speed = min(max(min(lims.values()), P['min_speed']), P['v_max'])
        self.last_info = dict(theta=theta, limiter=why, v_prof=v_prof, v_gov=v_gov, v_lat=v_lat, v_str=v_str,
                              d_arc=d_i, d_straight=d_straight, steer=steer, d_ftg=d_ftg)
        return speed, steer

    # ------------------------------------------------------------------
    # 1. Recovery state machine
    # ------------------------------------------------------------------
    def _enter_recovery(self, raw, ang, t):
        left = float(np.mean(raw[(ang > 0.8) & (ang < 1.7)]))
        right = float(np.mean(raw[(ang < -0.8) & (ang > -1.7)]))
        # which way is 'forward'? gap direction remembered from ~0.5 s before the trouble
        self.fwd_est = None
        for (th, psi, gap) in reversed(self.hist):
            if th <= t - 0.5:
                self.fwd_est = psi + gap
                break
        if self.fwd_est is None and self.hist:
            self.fwd_est = self.hist[0][1] + self.hist[0][2]
        err = self._heading_err()
        if 0.4 < abs(err) < 2.6:
            self.turn_dir = 1.0 if err > 0 else -1.0     # shortest way to track heading
        else:
            self.turn_dir = 1.0 if left >= right else -1.0   # toward more room
        self.rev_steer = -self.turn_dir * self.P['max_steer']
        self.psi_rec, self.t_prev, self.last_steer_cmd = 0.0, t, 0.0
        # never swing to full lock / reverse while still rolling fast: brake straight first
        self.mode = 'BRAKE' if abs(self.v) > 0.6 else 'REVERSE'
        self.mode_t0 = self.rec_t0 = t
        self.recoveries += 1

    # ------------------------------------------------------------------
    # trouble-spot memory
    # ------------------------------------------------------------------
    def _cell(self, s):
        return int(min(max(s, 0.0) // self.P['mem_bucket'], self.mem_n - 1))

    def mem_event(self, kind, steer, cell=None):
        """Remember a trouble spot: shift the preferred lane toward the OUTSIDE of the turn in the
        cells just before it, and (after a crash) slow the approach."""
        P = self.P
        b = self._cell(self.s) if cell is None else cell
        sg = 0.0 if abs(steer) < 0.08 else (1.0 if steer > 0 else -1.0)
        for k in range(int(P['mem_back']) + 1):
            i = b - k
            if i < 0:
                break
            w = 1.0 - 0.1 * k
            self.mem_lat[i] = float(np.clip(self.mem_lat[i] + P['mem_lat_step'] * w * sg, -1.0, 1.0))
            if kind == 'crash':
                self.mem_cap[i] = max(P['mem_cap_min'], self.mem_cap[i] * 0.92)
            self.mem_hit[i] = True

    def lap_reset(self):
        """Called at every finish-line crossing: re-zero the path length, relax old speed caps."""
        self.s = 0.0
        self.mem_cap = np.minimum(1.0, self.mem_cap + 0.02 * (~self.mem_hit))
        self.mem_hit[:] = False

    def mem_dump(self):
        return {'lat': [round(float(x), 3) for x in self.mem_lat],
                'cap': [round(float(x), 3) for x in self.mem_cap]}

    def mem_load(self, d):
        try:
            lat, cap = np.array(d['lat'], float), np.array(d['cap'], float)
            n = min(len(lat), self.mem_n)
            self.mem_lat[:n], self.mem_cap[:n] = lat[:n], cap[:n]
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def reset_state(self):
        """Forget everything about the previous situation (used after a teleport back to the line)."""
        self.mode = 'DRIVE'
        self.prev_steer = 0.0
        self.cmd_speed = 0.0
        self.last_out_steer = 0.0
        self.psi_dr = 0.0
        self.t_dr = None
        self.hist.clear()
        self.fwd_est = None
        self.in_contact = False
        self.crashed = False
        self.s = 0.0
        self._crawl_t = None
        self._crawl_fired = False
        self._blocked_t = None
        self._stuck_since = None
        self._spin_since = None

    def _heading_err(self):
        """Angle the car still has to turn (rad, +left) to face 'forward' again."""
        if self.P['use_odom_heading'] and self.ref is not None:
            return wrap(self.ref - self.yaw)
        if self.fwd_est is not None:
            return wrap(self.fwd_est - self.psi_dr)
        return 0.0

    def _exit_recovery(self):
        self.mode = 'DRIVE'
        self.prev_steer = 0.0
        self.cmd_speed = 0.0
        self._stuck_since = None

    def _recover(self, raw, ang, inc, t):
        P = self.P
        if t - self.rec_t0 > P['recovery_timeout']:
            self._exit_recovery()
            return self._plan(raw, ang, inc)

        if self.mode == 'BRAKE':
            if abs(self.v) < 0.5 or t - self.mode_t0 > P['brake_time']:
                self.mode, self.mode_t0 = 'REVERSE', t
                self.psi_rec, self.t_prev = 0.0, t
            else:
                self.last_steer_cmd = 0.0
                self.t_prev = t
                return 0.0, 0.0

        err = self._heading_err()
        # dead-reckon the rotation done during recovery: wheel speed + commanded steering
        dt = max(0.0, min(t - self.t_prev, 0.1))
        self.t_prev = t
        self.psi_rec += self.v * math.tan(self.last_steer_cmd) / P['wheelbase'] * dt
        if abs(self.psi_rec) > P['rot_limit']:
            # rotated far enough: never go round further (that is how the car ends up facing
            # backwards). Swing back the other way instead.
            if self.turn_dir * self.psi_rec > 0:
                self.turn_dir = -self.turn_dir
                self.rev_steer = -self.turn_dir * P['max_steer']
                self.mode, self.mode_t0 = 'REVERSE', t
        if self.mode == 'REVERSE':
            rear = raw[np.abs(ang) > 1.9]
            rear_min = float(np.min(rear)) if rear.size else 99.0
            if t - self.mode_t0 > P['rev_time'] or rear_min < P['rear_clear']:
                self.mode, self.mode_t0 = 'TURN', t
            else:
                self.last_steer_cmd = self.rev_steer
                return -P['rev_speed'], self.rev_steer

        steer = self.turn_dir * P['max_steer']
        if self._arc_clear(steer) < 0.35:                # blocked: back up again
            self.mode, self.mode_t0 = 'REVERSE', t
            self.last_steer_cmd = self.rev_steer
            return -P['rev_speed'], self.rev_steer
        if abs(err) < 0.5 and self._arc_clear(0.0) > 1.5:
            self._exit_recovery()
            return self._plan(raw, ang, inc)
        self.last_steer_cmd = steer
        return P['turn_speed'], steer

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def drive(self, raw, angle_min, inc, t):
        P = self.P
        raw = np.clip(np.nan_to_num(np.asarray(raw, dtype=np.float64),
                                    nan=P['max_range'], posinf=P['max_range'],
                                    neginf=0.0), 0.0, P['max_range'])
        if raw.size == 0 or not math.isfinite(inc) or inc <= 0.0:
            self.cmd_speed = 0.0
            self.last_out_steer = 0.0
            return 0.0, 0.0
        ang, dcar, cs, sn = self._beams(len(raw), angle_min, inc)
        if self.t_dr is not None:
            dt_dr = max(0.0, min(t - self.t_dr, 0.1))
            self.psi_dr += self.v * math.tan(self.last_out_steer) / P['wheelbase'] * dt_dr
            self.s += abs(self.v) * dt_dr
        self.t_dr = t

        v = raw < P['max_range'] - 0.05
        self._px, self._py = raw[v] * cs[v], raw[v] * sn[v]

        # contact = a beam shorter than the car body boundary (same rule as the gym)
        contact = bool(np.any(raw < dcar + 0.01))
        if contact:
            if not self.in_contact:
                self.collision_count += 1
            self.in_contact = True
        elif not np.any(raw < dcar + 0.05):
            self.in_contact = False

        if self.mode == 'DRIVE':
            stuck = False
            if self.cmd_speed > 0.5 and abs(self.v) < 0.1:
                if self._stuck_since is None:
                    self._stuck_since = t
                stuck = (t - self._stuck_since) > P['stuck_time']
            else:
                self._stuck_since = None
            # Compare measured yaw rate with the curvature we actually commanded. A single
            # odometry spike must not trip the crash handler; require sustained excess yaw.
            kin_cmd = abs(self.v) * abs(math.tan(self.last_out_steer)) / P['wheelbase']
            spin_limit = max(2.5, 1.8 * kin_cmd + P['spin_margin'])
            if abs(self.yaw_rate) > spin_limit:
                self._spin_since = self._spin_since or t
            else:
                self._spin_since = None
            spin = self._spin_since is not None and (t - self._spin_since) >= 0.15
            if not P['recovery']:
                if self.crashed:
                    return 0.0, 0.0
                # a crash is a crash: count it ONCE and let the node teleport the car back
                why = ('contact' if contact else 'spin' if spin else
                       'stuck' if stuck else 'wrong_way' if self.wrong_way else '')
                if why:
                    if not self.in_contact:
                        self.collision_count += 1
                    self.in_contact = True
                    self.crash_reason = why
                    self.crashed = True
                    return 0.0, 0.0
            elif contact or self.wrong_way or stuck or spin:
                self._enter_recovery(raw, ang, t)

        if self.mode == 'DRIVE':
            speed, steer = self._plan(raw, ang, inc)
            if P['memory'] and self.s > 8.0:
                if abs(self.v) < P['mem_crawl_speed']:
                    if self._crawl_t is None:
                        self._crawl_t, self._crawl_cell, self._crawl_fired = t, self._cell(self.s), False
                    elif not self._crawl_fired and t - self._crawl_t > P['mem_crawl_time']:
                        self._crawl_fired = True
                        self.mem_event('crawl', self.prev_steer, self._crawl_cell)
                else:
                    self._crawl_t = None
            if self.blocked:                    # no safe way forward
                # a single odd scan (wall corner grazing the safety margin) must not trigger a
                # reverse manoeuvre: require it to persist AND the car to be slow
                if self._blocked_t is None:
                    self._blocked_t = t
                if P['recovery'] and abs(self.v) < 1.5 and (t - self._blocked_t) > P['blocked_time']:
                    self._blocked_t = None
                    self._enter_recovery(raw, ang, t)
                    speed, steer = self._recover(raw, ang, inc, t)
            else:
                self._blocked_t = None
        else:
            speed, steer = self._recover(raw, ang, inc, t)
        self.cmd_speed = speed
        self.last_out_steer = steer
        if self.mode == 'DRIVE' and 'theta' in self.last_info:
            self.hist.append((t, self.psi_dr, self.last_info['theta']))
        return speed, steer


# ===========================================================================
# Learning: lap-by-lap "aggression" ramp (no map, no memory of the track)
# ===========================================================================
class AggressionLearner:
    """Lap-by-lap search for the highest safe "aggression" (no map, no track memory).

        aggr scales (a) the target speed and (b) the lateral-acceleration budget.

    * after a clean lap  : aggr += step, but never above `ceiling`
    * at a collision     : remember the failing value, drop to (best clean value - back_off),
                           and set ceiling = failing value - back_off
    * ceiling recovers   : after `retry_laps` clean laps AT the ceiling it is raised by step/2,
                           so one unlucky collision cannot keep the car slow forever
    mode 'learn' : the above, saved to learned_params.json
    mode 'race'  : load best_clean from the file and keep it fixed
    mode 'fixed' : use the aggr parameter as given
    """

    def __init__(self, mode, aggr0, step, back_off, a_min, a_max, path, retry_laps=3):
        self.mode, self.step, self.back_off = mode, step, back_off
        self.a_min, self.a_max, self.path, self.retry_laps = a_min, a_max, path, retry_laps
        self.aggr = aggr0
        self.ceiling = a_max
        self.best_clean = None          # highest aggr that gave a collision-free lap
        self.best_clean_time = None
        self.clean_at_ceiling = 0
        self.history = []

    def load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            bc = d.get('best_clean')
            self.aggr = float(bc if bc is not None else d['aggr'])
            self.best_clean = bc
            return True
        except (OSError, ValueError, KeyError):
            return False

    def save(self):
        with open(self.path, 'w') as f:
            json.dump({'aggr': self.aggr, 'best_clean': self.best_clean,
                       'best_clean_lap_time': self.best_clean_time,
                       'ceiling': self.ceiling, 'history': self.history}, f, indent=2)

    def on_collision(self):
        """Called the moment the LiDAR reports contact. Returns the new aggr."""
        if self.mode != 'learn':
            return self.aggr
        failed = self.aggr
        self.ceiling = max(self.a_min, failed - self.back_off)
        base = self.best_clean if self.best_clean is not None else failed - 2.0 * self.step
        self.aggr = max(self.a_min, min(base - self.back_off * 0.5, self.ceiling))
        self.clean_at_ceiling = 0
        self.save()
        return self.aggr

    def on_lap(self, lap_time, collisions):
        """Called at every finish-line crossing. Returns the new aggr."""
        self.history.append({'aggr': round(self.aggr, 3), 'lap_time': round(lap_time, 3),
                             'collisions': collisions})
        if self.mode == 'learn':
            if collisions == 0:
                if self.best_clean is None or self.aggr >= self.best_clean:
                    self.best_clean, self.best_clean_time = self.aggr, lap_time
                if self.aggr >= self.ceiling - 1e-9:
                    self.clean_at_ceiling += 1
                    if self.clean_at_ceiling >= self.retry_laps:      # retry a little higher
                        self.ceiling = min(self.a_max, self.ceiling + 0.5 * self.step)
                        self.clean_at_ceiling = 0
                self.aggr = min(self.a_max, self.ceiling, self.aggr + self.step)
            self.save()
        return self.aggr


# ===========================================================================
# ROS 2 node: IO, lap counter, lap timer, start/finish line, HUD, RViz markers
# ===========================================================================
class FollowTheGap(Node):
    def __init__(self):
        super().__init__('follow_the_gap')

        for k, v in PARAMS.items():
            self.declare_parameter(k, v)
        extra = {
            'scan_topic': '/scan', 'odom_topic': '/ego_racecar/odom',
            'drive_topic': '/drive',
            'target_laps': 10, 'arm_distance': 10.0, 'line_half_width': 3.0,
            'min_lap_time': 5.0, 'log_file': 'lap_log.csv',
            # Catalunya start pose = first centerline point, heading along the track
            'start_x': 0.0, 'start_y': 0.0, 'start_yaw': -2.14363,
            'reset_to_start': True,
            # after a collision the gym car just drives through the wall: put it back on the line
            'reset_on_collision': True,
            # learning
            'mode': 'learn',                 # learn | race | fixed
            'learn_step': 0.06, 'learn_back_off': 0.12,
            'race_margin': 0.08,             # race mode runs this far BELOW the best clean value
            'aggr_min': 0.6, 'aggr_max': 1.8,
            'params_file': 'learned_params.json',
            'memory_file': 'track_memory.json',
            'debug_file': 'ftg_debug.csv',     # '' = off. Send this file when something looks wrong.
            # visuals (cosmetic only: the driving code never reads this file)
            'centerline_file': 'Catalunya_centerline.csv',
        }
        for k, v in extra.items():
            self.declare_parameter(k, v)
        g = lambda k: self.get_parameter(k).value          # noqa: E731
        P = {k: g(k) for k in PARAMS}
        P['speed_dist'] = list(P['speed_dist'])
        P['speed_val'] = list(P['speed_val'])

        self.target_laps = int(g('target_laps'))
        self.arm_dist = g('arm_distance')
        self.line_hw = g('line_half_width')
        self.min_lap_time = g('min_lap_time')
        self.log_file = g('log_file')
        self.mode = str(g('mode')).lower()

        # ---- start / finish line ----
        sx, sy, syaw = g('start_x'), g('start_y'), g('start_yaw')
        self.fixed_start = not any(math.isnan(v) for v in (sx, sy, syaw))
        self.start = None            # (x, y, hx, hy)
        self.start_yaw = None
        if self.fixed_start:
            self.set_start(sx, sy, syaw)
        self.reset_to_start = bool(g('reset_to_start')) and self.fixed_start
        self.reset_on_collision = bool(g('reset_on_collision')) and self.fixed_start
        self.want_reset = self.reset_to_start
        self.reset_tries = 0
        self.attempt = 1

        self.driver = FTGDriver(P, ref_heading=self.start_yaw)

        # ---- learning ----
        self.learner = AggressionLearner(
            self.mode, float(P['aggr']), g('learn_step'), g('learn_back_off'),
            g('aggr_min'), g('aggr_max'), g('params_file'))
        if self.mode == 'race':
            if self.learner.load():
                self.learner.aggr = max(self.learner.a_min, self.learner.aggr - g('race_margin'))
                self.get_logger().info(f'RACE mode: best clean aggr minus margin = {self.learner.aggr:.2f}')
            else:
                self.get_logger().warn('RACE mode but no learned_params.json: using aggr param.')
        self.driver.P['aggr'] = self.learner.aggr
        self.memory_file = g('memory_file')
        if P['memory']:
            try:
                with open(self.memory_file) as f:
                    ok = self.driver.mem_load(json.load(f))
                self.get_logger().info(f'Trouble-spot memory loaded from {self.memory_file}' if ok else
                                       'Memory file unreadable: starting fresh.')
            except (OSError, ValueError):
                pass

        # ---- lap state ----
        self.ready = False           # True once the car is on the start line
        self.finished = False
        self.prev_s = 0.0
        self.armed = False
        self.laps = 0
        self.lap_start_t = None
        self.lap_times = []
        self.lap_collisions = 0
        self.lap_collision_log = []
        self.total_collisions = 0
        self.seen_collisions = 0
        self.consecutive_clean = 0

        # ---- telemetry for the HUD ----
        self.speed = 0.0
        self.accel = 0.0
        self.last_v, self.last_v_t = 0.0, None
        self.steer_cmd = 0.0
        self.trail = []                              # [(x, y, speed)]
        self.last_trail_xy = None

        # ---- ROS I/O ----
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, g('drive_topic'), 10)
        self.marker_pub = self.create_publisher(Marker, '/start_finish_line', latched)
        self.surface_pub = self.create_publisher(Marker, '/track_surface', latched)
        self.trail_pub = self.create_publisher(Marker, '/car_trail', 1)
        self.hud_pub = self.create_publisher(Image, '/race_hud', 1)
        self.hud_txt_pub = self.create_publisher(Marker, '/race_hud_text', 1)
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        # depth 1: if processing is ever slower than the scan rate, drop old scans instead of
        # queueing them (stale scans = delayed, 'drunk' steering)
        self.create_subscription(LaserScan, g('scan_topic'), self.scan_cb, QoSProfile(depth=1))
        self.create_subscription(Odometry, g('odom_topic'), self.odom_cb, 10)
        self.create_timer(1.0, self.status_cb)
        self.create_timer(1.0, self.publish_marker)
        self.create_timer(0.2, self.hud_cb)
        self.create_timer(0.1, self.trail_cb)
        self.surface_marker = self.build_surface(g('centerline_file'))
        self.create_timer(2.0, lambda: self.surface_marker and
                          self.surface_pub.publish(self.surface_marker))
        if self.fixed_start:
            self.create_timer(0.5, self.reset_cb)

        self.debug_fh, self.debug_w, self.debug_t = None, None, 0.0
        if g('debug_file'):
            self.debug_fh = open(g('debug_file'), 'w', newline='')
            self.debug_w = csv.writer(self.debug_fh)
            self.debug_w.writerow(['t', 'x', 'y', 'speed', 'cmd_speed', 'steer', 'mode', 'limiter',
                                   'd_arc', 'd_straight', 'aggr', 'collisions', 'recoveries',
                                   'compute_ms', 'lap', 'yaw_rate', 'crash'])
        self.slow_since = None
        self.get_logger().info(
            f'Follow-the-Gap (12 m/s cap, crash->teleport, lane centring, memory) started (mode={self.mode}, aggr={self.learner.aggr:.2f}). '
            f'Target: {self.target_laps} laps. LiDAR-only driving: '
            f'use_odom_heading={P["use_odom_heading"]}.')

    # ------------------------------------------------------------------
    def save_memory(self):
        if not self.driver.P['memory']:
            return
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.driver.mem_dump(), f)
        except OSError:
            pass

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def log_debug(self, speed, steer, ms):
        d, info = self.driver, self.driver.last_info
        t = self.now()
        # explain long crawls in the terminal
        if d.mode == 'DRIVE' and speed < 1.0:
            self.slow_since = self.slow_since or t
            if t - self.slow_since > 1.0:
                self.get_logger().warn(
                    f'Crawling ({speed:.1f} m/s) because: {info.get("limiter")} | '
                    f'free dist on chosen arc {info.get("d_arc", 0):.2f} m, straight '
                    f'{info.get("d_straight", 0):.2f} m', throttle_duration_sec=2.0)
        else:
            self.slow_since = None
        if self.debug_w and t - self.debug_t >= 0.1:
            self.debug_t = t
            xy = d.last_xy or (0.0, 0.0)
            self.debug_w.writerow([f'{t:.3f}', f'{xy[0]:.3f}', f'{xy[1]:.3f}', f'{self.speed:.3f}',
                                   f'{speed:.3f}', f'{steer:.3f}', d.mode, info.get('limiter', ''),
                                   f'{info.get("d_arc", 0):.2f}', f'{info.get("d_straight", 0):.2f}',
                                   f'{d.P["aggr"]:.3f}', self.total_collisions, d.recoveries,
                                   f'{ms:.1f}', self.laps + 1, f'{d.yaw_rate:.2f}', d.crash_reason])
            if int(t) % 5 == 0:
                self.debug_fh.flush()

    def publish_drive(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)
        self.steer_cmd = float(steer)

    def set_start(self, x, y, yaw):
        self.start = (x, y, math.cos(yaw), math.sin(yaw))
        self.start_yaw = yaw
        self.get_logger().info(
            f'Start/finish line at x={x:.2f}, y={y:.2f}, heading {math.degrees(yaw):.0f} deg.')

    # ---- RViz visuals ----------------------------------------------------------
    def publish_marker(self):
        """Checkered start/finish line + a 'START / FINISH' label."""
        if self.start is None:
            return
        sx, sy, _, _ = self.start
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'start_finish', 0
        m.type, m.action = Marker.CUBE_LIST, Marker.ADD
        sq = 0.4
        m.scale.x = m.scale.y = sq
        m.scale.z = 0.02
        m.pose.position.x, m.pose.position.y, m.pose.position.z = sx, sy, 0.03
        m.pose.orientation.z = math.sin(self.start_yaw / 2.0)
        m.pose.orientation.w = math.cos(self.start_yaw / 2.0)
        for i in range(int(2 * self.line_hw / sq)):
            lat = -self.line_hw + (i + 0.5) * sq
            for j in range(2):
                p = Point()
                p.x, p.y = (j - 0.5) * sq, lat
                m.points.append(p)
                c = ColorRGBA()
                c.r = c.g = c.b = 1.0 if (i + j) % 2 == 0 else 0.05
                c.a = 1.0
                m.colors.append(c)
        self.marker_pub.publish(m)

        t = Marker()
        t.header = m.header
        t.ns, t.id = 'start_finish', 1
        t.type, t.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = sx, sy, 1.2
        t.pose.orientation.w = 1.0
        t.scale.z = 0.9
        t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        t.text = 'START / FINISH'
        self.marker_pub.publish(t)

    def build_surface(self, path):
        """Dark-grey road surface from the centerline CSV (visual aid only)."""
        import os
        cands = [path, os.path.join(os.path.dirname(os.path.abspath(__file__)), path)]
        pts = None
        for c in cands:
            try:
                pts = np.loadtxt(c, delimiter=',', skiprows=1)
                break
            except (OSError, ValueError):
                continue
        if pts is None or pts.ndim != 2 or pts.shape[1] < 4:
            self.get_logger().info('No centerline CSV found: road surface not drawn (cosmetic).')
            return None
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id = 'track_surface', 0
        m.type, m.action = Marker.TRIANGLE_LIST, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.pose.orientation.w = 1.0
        m.color = ColorRGBA(r=0.22, g=0.24, b=0.28, a=1.0)
        n = len(pts)
        edges = []
        for i in range(n):
            j = (i + 1) % n
            d = pts[j, :2] - pts[i - 1, :2]
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            edges.append((pts[i, :2] + nrm * pts[i, 3], pts[i, :2] - nrm * pts[i, 2]))
        for i in range(n):
            a0, a1 = edges[i]
            b0, b1 = edges[(i + 1) % n]
            for tri in ((a0, a1, b0), (b0, a1, b1)):
                for q in tri:
                    p = Point()
                    p.x, p.y, p.z = float(q[0]), float(q[1]), -0.01
                    m.points.append(p)
        return m

    def trail_cb(self):
        """Car trail coloured by speed (blue = slow, red = fast): shows the line taken."""
        if not self.ready or self.finished or self.driver.yaw is None:
            return
        xy = self.driver.last_xy
        if xy is None:
            return
        self.trail.append((xy[0], xy[1], self.speed))
        self.trail = self.trail[-900:]
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'trail', 0
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.08
        m.pose.orientation.w = 1.0
        vmax = self.driver.P['v_max']
        for x, y, v in self.trail:
            p = Point()
            p.x, p.y, p.z = x, y, 0.05
            m.points.append(p)
            f = max(0.0, min(1.0, v / max(vmax, 1e-3)))
            m.colors.append(ColorRGBA(r=f, g=0.35 * (1 - f) + 0.2, b=1.0 - f, a=1.0))
        self.trail_pub.publish(m)

    def hud_cb(self):
        d = self.driver
        cur = (self.now() - self.lap_start_t) if (self.ready and not self.finished
                                                   and self.lap_start_t) else None
        state = 'FINISHED' if self.finished else ('RUNNING' if self.ready else 'WAITING')
        st = dict(laps=self.laps, target=self.target_laps, state=state, cur_time=cur,
                  lap_times=self.lap_times, lap_cols=self.lap_collision_log,
                  speed=abs(self.speed), accel=self.accel,
                  steer=self.steer_cmd, v_max=d.P['v_max'],
                  a_max=d.P['a_lat'], steer_max=d.P['max_steer'],
                  collisions=self.total_collisions, streak=self.consecutive_clean,
                  aggr=d.P['aggr'], mode=self.mode.upper())
        if HUD_OK:
            arr = hud_render(st)
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.height, msg.width = arr.shape[0], arr.shape[1]
            msg.encoding, msg.step = 'rgb8', arr.shape[1] * 3
            msg.data = arr.tobytes()
            self.hud_pub.publish(msg)
        else:                                       # Pillow missing: simple text in the 3D view
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = 'hud', 0
            m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            x, y = (d.last_xy or (0.0, 0.0))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 2.0
            m.pose.orientation.w = 1.0
            m.scale.z = 0.8
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lap_text = (f"LAP {self.laps + 1} (LEARN)" if self.mode == 'learn' else
                        f"LAP {min(self.laps + 1, self.target_laps)}/{self.target_laps}")
            m.text = (f"{lap_text}  {state}\nspeed {abs(self.speed):.1f} m/s  accel {self.accel:+.1f}\n"
                      + ''.join(f'L{i + 1} {t:.2f}s\n' for i, t in enumerate(self.lap_times[-5:])))
            self.hud_txt_pub.publish(m)

    def reset_cb(self):
        """Teleport the sim car onto the start line (gym listens to /initialpose)."""
        if not self.want_reset or self.ready or self.finished:
            return
        self.reset_tries += 1
        if self.reset_tries % 10 == 0:
            self.get_logger().warn(f'Still waiting for the teleport to the start line (try {self.reset_tries}).')
        sx, sy, _, _ = self.start
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = sx, sy
        msg.pose.pose.orientation.z = math.sin(self.start_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.start_yaw / 2.0)
        self.reset_pub.publish(msg)

    # ---- LiDAR -> command --------------------------------------------------
    def scan_cb(self, msg):
        if self.finished or not self.ready:
            self.publish_drive(0.0, 0.0)
            return
        t_start = time.perf_counter()
        speed, steer = self.driver.drive(
            msg.ranges, msg.angle_min, msg.angle_increment, self.now())
        ms = (time.perf_counter() - t_start) * 1e3
        if ms > 20.0:
            self.get_logger().warn(f'Slow controller step: {ms:.0f} ms (scan period may be shorter)',
                                   throttle_duration_sec=5.0)
        self.log_debug(speed, steer, ms)

        new = self.driver.collision_count - self.seen_collisions
        if new > 0:
            self.seen_collisions = self.driver.collision_count
            self.total_collisions += new
            self.lap_collisions += new
            self.consecutive_clean = 0
            old = self.driver.P['aggr']
            self.driver.P['aggr'] = self.learner.on_collision()
            self.get_logger().warn(
                f'COLLISION #{self.total_collisions} ({self.driver.crash_reason or "contact"}). Clean-lap streak reset. '
                f'aggr {old:.2f} -> {self.driver.P["aggr"]:.2f}')
            if self.driver.P['memory']:
                self.driver.mem_event('crash', self.driver.last_out_steer)
                self.save_memory()
            if self.reset_on_collision:
                self.restart_after_collision()
                self.publish_drive(0.0, 0.0)
                return
        self.publish_drive(speed, steer)

    # ---- odometry -> lap counter, lap timer, telemetry ---------------------
    # NOTE: the pose is used ONLY for scoring (lap counter / timer / HUD). With the default
    # use_odom_heading:=false the driving decisions come from LiDAR + wheel speed alone.
    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        v = msg.twist.twist.linear.x
        t = self.now()
        self.driver.update_pose(p.x, p.y, yaw, v, t, msg.twist.twist.angular.z)
        self.driver.last_xy = (p.x, p.y)
        self.speed = v
        if self.last_v_t is not None and t > self.last_v_t:
            a = (v - self.last_v) / (t - self.last_v_t)
            self.accel += 0.2 * (a - self.accel)         # low-pass filtered
        self.last_v, self.last_v_t = v, t
        if self.finished:
            return

        if self.start is None:                      # line = spawn pose
            self.set_start(p.x, p.y, yaw)
            self.driver.ref = yaw

        sx, sy, hx, hy = self.start
        dx, dy = p.x - sx, p.y - sy
        s = dx * hx + dy * hy                       # along the line normal
        lat = -dx * hy + dy * hx                    # along the line
        dist = math.hypot(dx, dy)

        if not self.ready:                          # wait until on the line
            if dist < 1.0 and (not self.want_reset or self.reset_tries >= 1):
                self.ready = True
                self.want_reset = False
                self.lap_start_t = t
                self.prev_s = s
                self.get_logger().info('Car is on the start line. GO!')
            return

        if not self.armed and dist > self.arm_dist:
            self.armed = True

        forward = math.cos(wrap(yaw - self.start_yaw)) > 0.0
        crossed = (self.armed and self.prev_s < 0.0 <= s and abs(lat) < self.line_hw
                   and forward)
        self.prev_s = s

        if crossed:
            lap_time = t - self.lap_start_t
            if lap_time < self.min_lap_time:
                return
            self.laps += 1
            self.lap_times.append(lap_time)
            self.lap_collision_log.append(self.lap_collisions)
            self.write_lap_log()
            self.driver.lap_reset()
            self.save_memory()
            if self.lap_collisions == 0:
                self.consecutive_clean += 1
            old = self.driver.P['aggr']
            self.driver.P['aggr'] = self.learner.on_lap(lap_time, self.lap_collisions)
            self.get_logger().info(
                f'LAP {self.laps}/{self.target_laps} | time {lap_time:.3f} s | '
                f'best {min(self.lap_times):.3f} s | '
                f'collisions this lap {self.lap_collisions} | '
                f'clean streak {self.consecutive_clean} | '
                f'aggr {old:.2f} -> {self.driver.P["aggr"]:.2f}')
            self.lap_start_t = t
            self.lap_collisions = 0
            self.armed = False
            if self.mode != 'learn' and self.laps >= self.target_laps:
                self.finish()

    def restart_after_collision(self):
        """Teleport back to the start line. learn mode keeps its lap history (the learner has
        already reacted); race/fixed mode restarts the 10-clean-lap attempt from zero."""
        if self.mode != 'learn':
            self.attempt += 1
            self.laps = 0
            self.lap_times, self.lap_collision_log = [], []
            self.get_logger().warn(f'Attempt failed. Starting attempt #{self.attempt}: '
                                   f'10 clean laps needed again.')
        self.lap_collisions = 0
        self.ready = False
        self.armed = False
        self.want_reset = True
        self.reset_tries = 0
        self.driver.reset_state()
        self.trail = []

    def status_cb(self):
        if self.finished or not self.ready:
            if not self.ready and not self.finished:
                self.get_logger().info('Waiting for the car to reach the start line...',
                                       throttle_duration_sec=5.0)
            return
        d = self.driver
        self.get_logger().info(
            f'Lap {self.laps + 1}: {self.now() - self.lap_start_t:.1f} s | '
            f'done {self.laps}/{self.target_laps} | mode {d.mode} | '
            f'speed {abs(self.speed):.1f} m/s | aggr {d.P["aggr"]:.2f} | '
            f'recoveries {d.recoveries}',
            throttle_duration_sec=4.0)

    def write_lap_log(self):
        # Persist after every lap so an indefinitely running learn session is still auditable.
        with open(self.log_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lap', 'lap_time_s', 'collisions_in_lap'])
            for i, (lt, c) in enumerate(zip(self.lap_times, self.lap_collision_log), 1):
                w.writerow([i, f'{lt:.3f}', c])

    def finish(self):
        self.finished = True
        self.publish_drive(0.0, 0.0)
        self.write_lap_log()
        log = self.get_logger().info
        log('=' * 50)
        log(f'FINISHED {self.laps} laps')
        for i, lt in enumerate(self.lap_times, 1):
            log(f'  Lap {i:2d}: {lt:.3f} s')
        log(f'Fastest lap : {min(self.lap_times):.3f} s')
        log(f'Collisions  : {self.total_collisions}')
        log(f'Recoveries  : {self.driver.recoveries}')
        log('Challenge 1 (10 clean laps): '
            + ('PASSED' if self.total_collisions == 0 else 'NOT passed'))
        log(f'Log saved to {self.log_file}')
        log('=' * 50)


def main():
    if not ROS_OK:
        raise SystemExit('ROS 2 (rclpy) not found. Source your ROS 2 setup.bash first.')
    rclpy.init()
    node = FollowTheGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_drive(0.0, 0.0)
        node.write_lap_log()
        node.learner.save() if node.mode == 'learn' else None
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

