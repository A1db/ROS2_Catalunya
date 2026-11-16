#!/usr/bin/env python3
"""
Race HUD renderer: draws a small dashboard (numpy RGB image) that the controller
publishes on /race_hud. In RViz add  Image -> Topic /race_hud  and you get a small
window with the lap counter, every lap time, speed, acceleration and steering.

No ROS imports here, so it can be tested on its own:
    python3 hud.py          # writes hud_preview.png
Needs Pillow (python3-pil), which ships with a normal ROS 2 desktop install.
"""
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HUD_OK = True
except ImportError:                      # the controller then falls back to a text marker
    HUD_OK = False

W, H = 480, 340
BG, PANEL = (18, 20, 26), (30, 34, 44)
WHITE, GREY = (235, 238, 245), (140, 148, 165)
GREEN, RED, AMBER, BLUE = (70, 220, 120), (240, 80, 80), (250, 190, 60), (90, 160, 255)

_FONTS = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        try:
            _FONTS[key] = ImageFont.truetype(name, size)
        except OSError:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def fmt_time(t):
    if t is None:
        return '--:--.---'
    m, s = divmod(t, 60.0)
    return f'{int(m)}:{s:06.3f}'


def _bar(d, x, y, w, h, frac, color, centered=False):
    d.rounded_rectangle([x, y, x + w, y + h], 4, fill=PANEL, outline=(60, 66, 82))
    frac = max(-1.0, min(1.0, frac))
    if centered:
        mid = x + w / 2
        x0, x1 = (mid, mid + frac * w / 2) if frac >= 0 else (mid + frac * w / 2, mid)
        d.rectangle([x0, y + 2, x1, y + h - 2], fill=color)
        d.line([mid, y, mid, y + h], fill=GREY)
    else:
        d.rectangle([x + 2, y + 2, x + 2 + max(0.0, frac) * (w - 4), y + h - 2], fill=color)


def render(s):
    """s: dict with keys laps, target, lap_times, lap_cols, cur_time, state, speed, accel,
    steer, v_max, a_max, steer_max, collisions, streak, aggr, mode."""
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)

    # ---- header: lap counter -------------------------------------------------
    d.text((16, 8), 'LAP', font=_font(16, True), fill=GREY)
    shown = s['laps'] + (0 if s['state'] == 'FINISHED' else 1)
    lap_label = f"{shown} (LEARN)" if s.get('mode', '').upper() == 'LEARN' else f"{shown}/{s['target']}"
    d.text((62, -2), lap_label, font=_font(32 if 'LEARN' in lap_label else 40, True), fill=WHITE)
    col = {'RUNNING': GREEN, 'FINISHED': BLUE, 'WAITING': AMBER}.get(s['state'], GREY)
    d.rounded_rectangle([W - 130, 14, W - 14, 40], 8, fill=col)
    d.text((W - 72, 27), s['state'], font=_font(14, True), fill=BG, anchor='mm')

    # ---- timers -----------------------------------------------------------------
    best = min(s['lap_times']) if s['lap_times'] else None
    last = s['lap_times'][-1] if s['lap_times'] else None
    d.text((16, 56), 'CURRENT', font=_font(11), fill=GREY)
    d.text((16, 68), fmt_time(s['cur_time']), font=_font(26, True), fill=WHITE)
    d.text((190, 56), 'LAST', font=_font(11), fill=GREY)
    d.text((190, 68), fmt_time(last), font=_font(26, True), fill=WHITE)
    d.text((345, 56), 'BEST', font=_font(11), fill=GREY)
    d.text((345, 68), fmt_time(best), font=_font(26, True), fill=GREEN if best else WHITE)

    # ---- lap table (2 columns x 5 rows) ---------------------------------------------
    d.line([16, 106, W - 16, 106], fill=(60, 66, 82))
    recent_times = s['lap_times'][-10:]
    recent_cols = s['lap_cols'][-10:]
    for i in range(10):
        cx, cy = 16 + (i // 5) * 232, 114 + (i % 5) * 19
        lap_number = max(1, s['laps'] - len(recent_times) + i + 1)
        d.text((cx, cy), f'{lap_number:>2}', font=_font(13, True), fill=GREY)
        if i < len(recent_times):
            t, c = recent_times[i], recent_cols[i]
            colr = RED if c else (GREEN if t == best else WHITE)
            d.text((cx + 28, cy), fmt_time(t), font=_font(13), fill=colr)
            if c:
                d.text((cx + 130, cy), f'{c} hit', font=_font(12), fill=RED)
        else:
            d.text((cx + 28, cy), '--:--.---', font=_font(13), fill=(70, 76, 92))

    # ---- speed / accel / steering ----------------------------------------------------
    d.line([16, 214, W - 16, 214], fill=(60, 66, 82))
    d.text((16, 222), 'SPEED', font=_font(11), fill=GREY)
    d.text((100, 218), f"{s['speed']:.2f} m/s  ({s['speed'] * 3.6:.0f} km/h)",
           font=_font(16, True), fill=WHITE)
    _bar(d, 16, 242, W - 32, 14, s['speed'] / s['v_max'], BLUE)

    d.text((16, 264), 'ACCEL', font=_font(11), fill=GREY)
    d.text((100, 260), f"{s['accel']:+.1f} m/s²", font=_font(14, True),
           fill=GREEN if s['accel'] >= 0 else RED)
    _bar(d, 200, 262, W - 216, 12, s['accel'] / s['a_max'],
         GREEN if s['accel'] >= 0 else RED, centered=True)
    d.text((16, 284), 'STEER', font=_font(11), fill=GREY)
    d.text((100, 280), f"{np.degrees(s['steer']):+.0f}°", font=_font(14, True), fill=WHITE)
    _bar(d, 200, 282, W - 216, 12, s['steer'] / s['steer_max'], AMBER, centered=True)

    # ---- footer ---------------------------------------------------------------------------
    d.line([16, 304, W - 16, 304], fill=(60, 66, 82))
    hit = RED if s['collisions'] else GREEN
    d.text((16, 312), f"hits {s['collisions']}", font=_font(13, True), fill=hit)
    d.text((100, 312), f"clean streak {s['streak']}", font=_font(13), fill=WHITE)
    d.text((250, 312), f"aggr {s['aggr']:.2f}", font=_font(13), fill=WHITE)
    d.text((340, 312), s['mode'], font=_font(13, True), fill=AMBER)
    return np.asarray(img, dtype=np.uint8)


if __name__ == '__main__':
    st = dict(laps=4, target=10, state='RUNNING', cur_time=31.4,
              lap_times=[84.12, 79.31, 76.88, 77.40], lap_cols=[0, 1, 0, 0],
              speed=5.62, accel=1.8, steer=-0.12, v_max=8.0, a_max=7.0, steer_max=0.4189,
              collisions=1, streak=2, aggr=1.15, mode='LEARN')
    Image.fromarray(render(st)).save('hud_preview.png')
    print('wrote hud_preview.png')

