# Génère le logo LaguineeDje (SVG + PNG). Dépendances : fonttools shapely scipy numpy cairosvg.
# Police : Anton (OFL, Google Fonts) à placer dans fonts/Anton-Regular.ttf. Usage : python3 build_logo.py <dossier_sortie> [TEXTE]  (ex. LaguineeDje pour la version en minuscules)
import sys, numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity

NAVY, TURQ = "#082B49", "#00A6A6"
font = TTFont("fonts/Anton-Regular.ttf")
gs, cmap, upm = font.getGlyphSet(), font.getBestCmap(), font["head"].unitsPerEm
CAP = font["OS/2"].sCapHeight

def glyph_contours(ch):
    pen = DecomposingRecordingPen(gs); gs[cmap[ord(ch)]].draw(pen)
    cs, cur = [], []
    for op, a in pen.value:
        if op == "moveTo": cur = [a[0]]
        elif op == "lineTo": cur.append(a[0])
        elif op == "qCurveTo":
            pts = list(a); p0 = cur[-1]
            # expand implied on-curve points
            segs = []
            for i in range(len(pts) - 1):
                c = pts[i]
                e = pts[i+1] if i == len(pts) - 2 else ((pts[i][0]+pts[i+1][0])/2, (pts[i][1]+pts[i+1][1])/2)
                segs.append((c, e))
            for c, e in segs:
                for t in np.linspace(0, 1, 13)[1:]:
                    cur.append(((1-t)**2*p0[0]+2*(1-t)*t*c[0]+t*t*e[0], (1-t)**2*p0[1]+2*(1-t)*t*c[1]+t*t*e[1]))
                p0 = e
        elif op == "curveTo":
            p0 = cur[-1]; c1, c2, e = a
            for t in np.linspace(0, 1, 17)[1:]:
                cur.append(tuple((1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(c1)+3*(1-t)*t*t*np.array(c2)+t**3*np.array(e)))
        elif op in ("closePath", "endPath"):
            if len(cur) > 2: cs.append(cur)
            cur = []
    return cs, gs[cmap[ord(ch)]].width

def densify(c, step):
    out = []
    for i in range(len(c)):
        a, b = np.array(c[i]), np.array(c[(i+1) % len(c)])
        n = max(1, int(np.hypot(*(b-a)) / step))
        for k in range(n): out.append(a + (b-a)*k/n)
    return out

def layout(text, track=0, xscale=None):
    """returns list (per glyph) of contour lists in font units, total width"""
    x, res = 0, []
    for i, ch in enumerate(text):
        if ch == " ": x += 0.28*upm + track; continue
        cs, w = glyph_contours(ch)
        s = (xscale or {}).get(i, 1)
        res.append([[(x + px*s, py) for px, py in c] for c in cs])
        x += w*s + track
    return res, x - track

def to_poly(contours):
    g = None
    for c in sorted(contours, key=lambda c: -abs(Polygon(c).area)):
        p = Polygon(c).buffer(0)
        g = p if g is None else g.symmetric_difference(p)
    return g

# ---------- fish envelope ----------
W, HH = 1000.0, 122.0
tu = [0, .03, .08, .16, .26, .38, .52, .68, .80, .885, .93, .965, 1.0]
tv = [.0, .17, .38, .64, .86, 1.0, .96, .78, .58, .44, .48, .70, 1.0]
bv = [.0, .15, .33, .56, .78, .92, .90, .76, .58, .44, .48, .70, 1.0]
top = PchipInterpolator(tu, np.array(tv)*HH)
bot = PchipInterpolator(tu, -np.array(bv)*HH)

TEXT = sys.argv[2] if len(sys.argv) > 2 else "LAGUINEEDJE"
MIXED = TEXT != TEXT.upper()   # casse mixte : chaque lettre est étirée pour remplir la silhouette
glyphs, tw = layout(TEXT, track=-6, xscale={0: 1.35, len(TEXT)-1: 1.8 if MIXED else 1.5})
U0 = 0.105
def warp(p, y0=0, y1=CAP):
    u = U0 + (1-U0)*np.clip(p[0]/tw, 0, 1); v = (p[1]-y0)/(y1-y0)
    x = u*W
    return (x, bot(u) + v*(top(u)-bot(u)))
parts = []
for g in glyphs:
    ys = [q[1] for c in g for q in c]
    y0, y1 = (min(ys), max(ys)) if MIXED else (0, CAP)
    parts.append(to_poly([[warp(q, y0, y1) for q in densify(c, 25)] for c in g]))
body = unary_union(parts)

def blade(pts, n=24):
    """closed shape from quadratic segments: list of (p0, ctrl, p1)"""
    out = []
    for p0, c, p1 in pts:
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(((1-t)**2*p0[0]+2*(1-t)*t*c[0]+t*t*p1[0], (1-t)**2*p0[1]+2*(1-t)*t*c[1]+t*t*p1[1]))
    return Polygon(out).buffer(0)

T = lambda u: float(top(u)); B = lambda u: float(bot(u))
def fin(i, f0, f1, sweep, h, up=True, inset=14):
    """fin rooted on letter i, between fractions f0..f1 of its width"""
    lx0, _, lx1, _ = parts[i].bounds
    xa, xb = lx0+(lx1-lx0)*f0, lx0+(lx1-lx0)*f1
    E = (lambda x: T(x/W)-inset) if up else (lambda x: B(x/W)+inset)
    sgn = 1 if up else -1
    a, b = (xa, E(xa)), (xb, E(xb)); tip = (xb+sweep, E(xb)+sgn*(h+inset))
    return blade([(a, (a[0]+(tip[0]-a[0])*.62, a[1]+(tip[1]-a[1])*.30), tip),
                  (tip, (b[0]+sweep*.10, b[1]+(tip[1]-b[1])*.30), b),
                  (b, ((a[0]+b[0])/2, E((xa+xb)/2)-sgn*4), a)])
fins = [fin(1, .15, 1.0, 26, 62), fin(6, .0, 1.0, 30, 40),
        fin(3, .0, 1.0, 34, 48, up=False), fin(7, .0, 1.0, 30, 36, up=False)]
xe, yt, yb = W, T(1.0), B(1.0)
_ys = sorted({round(p[1]) for c in glyph_contours("E")[0] for p in c})
arm_t, arm_b = (CAP-_ys[-2])/CAP*(yt-yb), (_ys[1]-_ys[0])/CAP*(yt-yb)
def quad(p0, c, p1, n=30):
    return [((1-t)**2*p0[0]+2*(1-t)*t*c[0]+t*t*p1[0], (1-t)**2*p0[1]+2*(1-t)*t*c[1]+t*t*p1[1]) for t in np.linspace(0, 1, n)]
LOBE_ARM = 0.2
def lobe(s):
    f = top if s > 0 else bot
    arm = arm_t if s > 0 else arm_b
    if MIXED: arm = (yt-yb)*LOBE_ARM
    y = float(f(1.0)); sl = float(f.derivative()(1.0))/W
    xi = xe-30; yi = float(f(xi/W))
    a, tip, c = (xe, y), (xe+150, y+s*72), (xe, y-s*arm)
    pts = [(xi, yi-s*3)] + quad(a, (xe+50, y+50*sl), tip) + quad(tip, (xe+55, y-s*arm*.15), c) + [(xi, yi-s*arm*.8)]
    return Polygon(pts).buffer(0)
tail_up, tail_dn = lobe(1), lobe(-1)
us = np.linspace(0, U0+0.012, 40)
snout = Polygon([(u*W, T(u)) for u in us] + [(u*W, B(u)) for u in us[::-1]]).buffer(0)
fish = unary_union([body, snout, tail_up, tail_dn] + fins)

# cuts: eye, mouth, slashes
eye = Point(88, 24).buffer(8.5)
mouth = blade([((-5, -9), (45, -22), (100, -14)), ((100, -14), (45, -12), (-5, -2))])
def slash(x0, y0, x1, y1, w):
    mx, my = (x0+x1)/2, (y0+y1)/2; dx, dy = x1-x0, y1-y0; L = np.hypot(dx, dy); nx, ny = -dy/L*w, dx/L*w
    return blade([((x0, y0), (mx+nx, my+ny), (x1, y1)), ((x1, y1), (mx-nx*.15, my-ny*.15), (x0, y0))])
cuts = [eye, mouth,
        slash(345, T(.33)+30, 285, -25, 9), slash(520, B(.50)-30, 585, 30, 9),
        slash(760, T(.74)+25, 715, -5, 7)]
if MIXED:  # entailles placées sur les fûts épais (u, n, D) pour ne pas abîmer les lettres fines
    def on(i, f, up, w=7):
        x0, _, x1, _ = parts[i].bounds; x = x0+(x1-x0)*f
        return slash(x+22, T(x/W)+25, x-18, 10, w) if up else slash(x-22, B(x/W)-25, x+18, -10, w)
    cuts = [eye, mouth, on(3, .30, True), on(5, .28, False), on(8, .30, True, 6)]
fish = fish.difference(unary_union(cuts))

# ---------- tagline ----------
tg, ttw = layout("MANGER MIEUX", track=0.42*upm)
ts = 64/CAP
tag = unary_union([to_poly(g) for g in tg])
tag = affinity.scale(tag, ts, ts, origin=(0, 0))
fx0, fy0, fx1, fy1 = fish.bounds
tag = affinity.translate(tag, (fx0+fx1)/2 - ttw*ts/2, fy0 - 64 - 46)

def path(geom):
    d = []
    for poly in getattr(geom, "geoms", [geom]):
        for ring in [poly.exterior] + list(poly.interiors):
            c = list(ring.coords)[:-1]
            d.append("M" + "L".join(f"{x:.1f} {-y:.1f}" for x, y in c) + "Z")
    return "".join(d)

fish = fish.simplify(0.15); tag = tag.simplify(0.05)
x0 = min(fish.bounds[0], tag.bounds[0]); x1 = max(fish.bounds[2], tag.bounds[2])
y0 = tag.bounds[1]; y1 = fish.bounds[3]
def svg(bg, pad):
    vb = f"{x0-pad:.0f} {-y1-pad:.0f} {x1-x0+2*pad:.0f} {y1-y0+2*pad:.0f}"
    r = f'<rect x="{x0-pad:.0f}" y="{-y1-pad:.0f}" width="100%" height="100%" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" role="img" aria-label="LaguineeDje — Manger mieux">'
            f'<title>LaguineeDje — Manger mieux</title>{r}'
            f'<path fill="{NAVY}" fill-rule="evenodd" d="{path(fish)}"/>'
            f'<path fill="{TURQ}" fill-rule="evenodd" d="{path(tag)}"/></svg>')
out = sys.argv[1] if len(sys.argv) > 1 else "out"
if MIXED: out += "/logo-laguineedje-minuscules"
else: out += "/logo-laguineedje"
open(f"{out}.svg", "w").write(svg(None, 40))
import cairosvg
cairosvg.svg2png(bytestring=svg("#FFFFFF", 150).encode(), write_to=f"{out}.png", output_width=2400)
cairosvg.svg2png(bytestring=svg(None, 40).encode(), write_to=f"{out}-transparent.png", output_width=2400)
