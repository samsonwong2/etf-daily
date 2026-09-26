"""Plotly HTML helpers for adaptive regime charts."""
from __future__ import annotations

from pathlib import Path

# Injected after Plotly.newPlot.
# - Toolbar: start-date / end-date inputs (also ?start-date=&end-date=)
# - Box-zoom / pan / date apply: rescale every Y axis to the visible X window
Y_AUTOSCALE_ON_X_ZOOM_JS = r"""
(function () {
  var gd = document.querySelectorAll('.js-plotly-plot');
  if (!gd || !gd.length) return;
  var plot = gd[gd.length - 1];
  var PAD = 0.06;
  var timer = null;
  var syncing = false;
  var startEl = null;
  var endEl = null;

  function pad2(n) {
    return (n < 10 ? '0' : '') + n;
  }

  function toMs(v) {
    if (v == null) return NaN;
    if (typeof v === 'number') return v;
    if (v instanceof Date) return v.getTime();
    var s = String(v);
    // Pandas datetime64[ns] serializes as 2019-12-05T00:00:00.000000000;
    // Date.parse rejects 9-digit nanos. Always take the calendar day as UTC.
    var m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) {
      return Date.UTC(+m[1], +m[2] - 1, +m[3]);
    }
    var t = Date.parse(s);
    return isFinite(t) ? t : NaN;
  }

  function toYmd(v) {
    if (v == null) return '';
    if (typeof v === 'string') {
      var m = v.match(/^(\d{4}-\d{2}-\d{2})/);
      if (m) return m[1];
    }
    var t = toMs(v);
    if (!isFinite(t)) return '';
    var d = new Date(t);
    return d.getUTCFullYear() + '-' + pad2(d.getUTCMonth() + 1) + '-' + pad2(d.getUTCDate());
  }

  function unpackBdata(obj) {
    if (!obj || !obj.bdata) return null;
    if (obj._unpacked) return obj._unpacked;
    var bin;
    try {
      bin = atob(obj.bdata);
    } catch (err) {
      return null;
    }
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    var dtype = obj.dtype || 'f8';
    var size = dtype === 'f4' || dtype === 'i4' ? 4 : 8;
    var n = Math.floor(bytes.length / size);
    var view = new DataView(bytes.buffer, bytes.byteOffset, n * size);
    var out = new Array(n);
    var k;
    for (k = 0; k < n; k++) {
      if (dtype === 'f4') out[k] = view.getFloat32(k * 4, true);
      else if (dtype === 'i4') out[k] = view.getInt32(k * 4, true);
      else out[k] = view.getFloat64(k * 8, true);
    }
    obj._unpacked = out;
    return out;
  }

  function asArray(v) {
    if (v == null) return null;
    if (typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView(v)) return v;
    if (Array.isArray(v)) return v;
    if (typeof v === 'object' && v.bdata) return unpackBdata(v);
    return null;
  }

  function traceList() {
    if (plot._fullData && plot._fullData.length) return plot._fullData;
    return plot.data || [];
  }

  function xRange(layout) {
    var xa = layout.xaxis || {};
    if (xa.range && xa.range.length === 2) {
      return [toMs(xa.range[0]), toMs(xa.range[1])];
    }
    return null;
  }

  function inX(xv, lo, hi) {
    var t = toMs(xv);
    return isFinite(t) && t >= lo && t <= hi;
  }

  function pushVals(arr, v) {
    if (v == null) return;
    if (Array.isArray(v)) {
      for (var i = 0; i < v.length; i++) pushVals(arr, v[i]);
      return;
    }
    var n = +v;
    if (isFinite(n)) arr.push(n);
  }

  function extent(vals) {
    if (!vals.length) return null;
    var mn = Math.min.apply(null, vals);
    var mx = Math.max.apply(null, vals);
    if (!(isFinite(mn) && isFinite(mx))) return null;
    if (mn === mx) {
      var eps = Math.max(Math.abs(mn) * 0.01, 1e-6);
      return [mn - eps, mx + eps];
    }
    var pad = (mx - mn) * PAD;
    return [mn - pad, mx + pad];
  }

  function axisKey(yaxis) {
    if (!yaxis || yaxis === 'y') return 'yaxis';
    return 'yaxis' + String(yaxis).replace(/^y/, '');
  }

  function dataXBoundsYmd() {
    var mn = Infinity;
    var mx = -Infinity;
    traceList().forEach(function (tr) {
      var xs = tr && asArray(tr.x);
      if (!xs || !xs.length) return;
      for (var i = 0; i < xs.length; i++) {
        var t = toMs(xs[i]);
        if (!isFinite(t)) continue;
        if (t < mn) mn = t;
        if (t > mx) mx = t;
      }
    });
    if (!isFinite(mn) || !isFinite(mx)) return null;
    return [toYmd(mn), toYmd(mx)];
  }

  function visibleYmd() {
    var layout = plot.layout || {};
    var xa = layout.xaxis || {};
    if (xa.range && xa.range.length === 2) {
      var a = toYmd(xa.range[0]);
      var b = toYmd(xa.range[1]);
      if (a && b) return [a, b];
    }
    return dataXBoundsYmd();
  }

  function xAxisLayoutKeys() {
    var keys = ['xaxis'];
    var layout = plot.layout || {};
    Object.keys(layout).forEach(function (k) {
      if (/^xaxis\d+$/.test(k)) keys.push(k);
    });
    return keys;
  }

  function hasRealX(xs) {
    if (!xs || !xs.length) return false;
    for (var i = 0; i < xs.length; i++) {
      if (xs[i] != null && isFinite(toMs(xs[i]))) return true;
    }
    return false;
  }

  function uniqueFiniteYs(ys) {
    var seen = {};
    var out = [];
    if (!ys) return out;
    for (var i = 0; i < ys.length; i++) {
      var v = +ys[i];
      if (!isFinite(v)) continue;
      var key = String(v);
      if (seen[key]) continue;
      seen[key] = true;
      out.push(v);
    }
    return out;
  }

  function findCandleTrace(data) {
    for (var i = 0; i < data.length; i++) {
      var tr = data[i];
      if (tr && (tr.type || '') === 'candlestick') return { tr: tr, idx: i };
    }
    return null;
  }

  function buildCandleMaps(candle) {
    var xs = asArray(candle.x);
    var lows = asArray(candle.low);
    var highs = asArray(candle.high);
    var byDay = {};
    var gLo = Infinity;
    var gHi = -Infinity;
    if (!xs) return { byDay: byDay, gLo: gLo, gHi: gHi };
    for (var i = 0; i < xs.length; i++) {
      var t = toMs(xs[i]);
      if (!isFinite(t)) continue;
      var day = toYmd(t);
      var loV = lows ? +lows[i] : NaN;
      var hiV = highs ? +highs[i] : NaN;
      byDay[day] = { lo: loV, hi: hiV, t: t };
      if (isFinite(loV) && loV < gLo) gLo = loV;
      if (isFinite(hiV) && hiV > gHi) gHi = hiV;
    }
    return { byDay: byDay, gLo: gLo, gHi: gHi };
  }

  function isPricePanelMarker(tr) {
    if (!tr || (tr.type && tr.type !== 'scatter')) return false;
    var ya = tr.yaxis || 'y';
    if (ya !== 'y') return false;
    var mode = String(tr.mode || '');
    if (mode.indexOf('markers') < 0) return false;
    var xs = asArray(tr.x);
    return hasRealX(xs);
  }

  // Overlay triangles/diamonds sit on full-history Y rails. After an X date
  // window, candle-only Y autoscale clips those rails off the subplot — remap
  // them onto local rails (and near-candle trade marks onto local offsets).
  function remapPriceMarkers(lo, hi, byAxis) {
    var data = plot.data || [];
    var found = findCandleTrace(data);
    if (!found) return;
    var maps = buildCandleMaps(found.tr);
    if (!isFinite(maps.gLo) || !isFinite(maps.gHi)) return;
    var gMid = (maps.gLo + maps.gHi) / 2;

    var visLo = Infinity;
    var visHi = -Infinity;
    Object.keys(maps.byDay).forEach(function (day) {
      var c = maps.byDay[day];
      if (!c || !inX(c.t, lo, hi)) return;
      if (isFinite(c.lo) && c.lo < visLo) visLo = c.lo;
      if (isFinite(c.hi) && c.hi > visHi) visHi = c.hi;
    });
    if (!isFinite(visLo) || !isFinite(visHi)) return;
    var span = Math.max(visHi - visLo, Math.abs(visHi) * 0.02, 1e-6);
    var railAbove = visHi + span * 0.10;
    var railBelow = visLo - span * 0.10;
    var railGap = span * 0.045;
    var nearOff = span * 0.035;

    var railTraces = [];
    var restyleIdx = [];
    var restyleY = [];

    data.forEach(function (tr, idx) {
      if (!isPricePanelMarker(tr)) return;
      var xs = asArray(tr.x);
      var ys = asArray(tr.y);
      if (!xs || !ys) return;
      if (!tr._origY) {
        tr._origY = Array.prototype.slice.call(ys);
      }
      var orig = tr._origY;
      var uniq = uniqueFiniteYs(orig);
      if (uniq.length === 1) {
        railTraces.push({
          idx: idx,
          y0: uniq[0],
          n: Math.min(xs.length, orig.length),
          orig: orig,
        });
        return;
      }
      // Per-day marks (上涨买/卖, 极端跌幅, …): pin to local candle offset.
      var newY = new Array(orig.length);
      var changed = false;
      for (var i = 0; i < orig.length; i++) {
        var y0 = +orig[i];
        newY[i] = orig[i];
        if (!isFinite(y0)) continue;
        var day = toYmd(xs[i]);
        var c = maps.byDay[day];
        if (!c || !isFinite(c.lo) || !isFinite(c.hi)) continue;
        if (y0 < c.lo - 1e-12) {
          newY[i] = c.lo - nearOff;
          changed = true;
        } else if (y0 > c.hi + 1e-12) {
          newY[i] = c.hi + nearOff;
          changed = true;
        }
        if (inX(xs[i], lo, hi) && isFinite(+newY[i])) {
          pushVals(byAxis.yaxis.vals, newY[i]);
        }
      }
      if (changed) {
        restyleIdx.push(idx);
        restyleY.push(newY);
      }
    });

    railTraces.sort(function (a, b) {
      return a.y0 - b.y0;
    });
    var above = [];
    var below = [];
    railTraces.forEach(function (item) {
      if (item.y0 >= gMid) above.push(item);
      else below.push(item);
    });
    above.sort(function (a, b) {
      return a.y0 - b.y0;
    });
    below.sort(function (a, b) {
      return b.y0 - a.y0;
    });

    function assignRail(items, base, sign) {
      for (var s = 0; s < items.length; s++) {
        var item = items[s];
        var yRail = base + sign * s * railGap;
        var newY = new Array(item.n);
        for (var i = 0; i < item.n; i++) newY[i] = yRail;
        restyleIdx.push(item.idx);
        restyleY.push(newY);
        // Extent uses visible x only; y is constant so one sample is enough.
        var xs = asArray(data[item.idx].x);
        if (!xs) continue;
        for (var j = 0; j < xs.length; j++) {
          if (!inX(xs[j], lo, hi)) continue;
          pushVals(byAxis.yaxis.vals, yRail);
          break;
        }
      }
    }
    assignRail(above, railAbove, 1);
    assignRail(below, railBelow, -1);

    if (restyleIdx.length && typeof Plotly !== 'undefined' && Plotly.restyle) {
      Plotly.restyle(plot, { y: restyleY }, restyleIdx);
    }
  }

  function rescale() {
    var layout = plot.layout || {};
    var xr = xRange(layout);
    if (!xr || !isFinite(xr[0]) || !isFinite(xr[1])) return;
    var lo = Math.min(xr[0], xr[1]);
    var hi = Math.max(xr[0], xr[1]);
    var byAxis = {};

    var traces = traceList();
    traces.forEach(function (tr) {
      var xs = tr && asArray(tr.x);
      if (!xs || !xs.length) return;
      var ya = tr.yaxis || 'y';
      var key = axisKey(ya);
      if (!byAxis[key]) byAxis[key] = { candle: false, vals: [] };
      var bucket = byAxis[key];
      var typ = tr.type || 'scatter';

      if (typ === 'candlestick') {
        bucket.candle = true;
        var lows = asArray(tr.low);
        var highs = asArray(tr.high);
        var opens = asArray(tr.open);
        var closes = asArray(tr.close);
        for (var i = 0; i < xs.length; i++) {
          if (!inX(xs[i], lo, hi)) continue;
          pushVals(bucket.vals, lows && lows[i]);
          pushVals(bucket.vals, highs && highs[i]);
          pushVals(bucket.vals, opens && opens[i]);
          pushVals(bucket.vals, closes && closes[i]);
        }
      }
    });

    if (!byAxis.yaxis) byAxis.yaxis = { candle: false, vals: [] };
    if (byAxis.yaxis.candle) {
      remapPriceMarkers(lo, hi, byAxis);
    }

    traces.forEach(function (tr) {
      var xs = tr && asArray(tr.x);
      if (!xs || !xs.length) return;
      var ya = tr.yaxis || 'y';
      var key = axisKey(ya);
      if (!byAxis[key]) byAxis[key] = { candle: false, vals: [] };
      var bucket = byAxis[key];
      if (bucket.candle) return;
      var typ = tr.type || 'scatter';
      if (typ === 'candlestick') return;

      var ys = asArray(tr.y);
      if (!ys || !ys.length) return;

      var allNull = true;
      for (var j = 0; j < xs.length; j++) {
        if (xs[j] != null) {
          allNull = false;
          break;
        }
      }
      if (allNull) return;

      var nHit = 0;
      var n = Math.min(xs.length, ys.length);
      for (var k = 0; k < n; k++) {
        if (!inX(xs[k], lo, hi)) continue;
        nHit += 1;
        pushVals(bucket.vals, ys[k]);
      }
      // Horizontal ref lines (ERP 4.3% / 差额=0) only store two endpoints.
      // If they span the window, keep those y values so 0 stays on the gap axis.
      if (!nHit && n >= 2) {
        var t0 = toMs(xs[0]);
        var t1 = toMs(xs[n - 1]);
        if (isFinite(t0) && isFinite(t1)) {
          var a = Math.min(t0, t1);
          var b = Math.max(t0, t1);
          if (a <= hi && b >= lo) {
            pushVals(bucket.vals, ys[0]);
            pushVals(bucket.vals, ys[n - 1]);
          }
        }
      }
    });

    var update = {};
    Object.keys(byAxis).forEach(function (key) {
      var ext = extent(byAxis[key].vals);
      if (!ext) return;
      update[key + '.range'] = ext;
      update[key + '.autorange'] = false;
    });
    if (Object.keys(update).length) {
      Plotly.relayout(plot, update);
    }
  }

  function schedule() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () {
      timer = null;
      rescale();
    }, 60);
  }

  function syncInputsFromPlot() {
    if (!startEl || !endEl) return;
    var pair = visibleYmd();
    if (!pair) return;
    syncing = true;
    startEl.value = pair[0];
    endEl.value = pair[1];
    syncing = false;
  }

  function writeUrlDates(s, e) {
    try {
      var u = new URL(window.location.href);
      u.searchParams.set('start-date', s);
      u.searchParams.set('end-date', e);
      history.replaceState(null, '', u.toString());
    } catch (err) {}
  }

  function readUrlDates() {
    try {
      var q = new URLSearchParams(window.location.search);
      return [q.get('start-date') || q.get('start'), q.get('end-date') || q.get('end')];
    } catch (err) {
      return [null, null];
    }
  }

  function applyDateWindow(startYmd, endYmd, fromUrl) {
    if (!startYmd || !endYmd) return;
    if (startYmd > endYmd) {
      var tmp = startYmd;
      startYmd = endYmd;
      endYmd = tmp;
    }
    var bounds = dataXBoundsYmd();
    if (bounds) {
      if (startYmd < bounds[0]) startYmd = bounds[0];
      if (endYmd > bounds[1]) endYmd = bounds[1];
    }
    var update = {};
    xAxisLayoutKeys().forEach(function (key) {
      update[key + '.range'] = [startYmd + ' 00:00:00.000', endYmd + ' 23:59:59.999'];
      update[key + '.autorange'] = false;
    });
    Plotly.relayout(plot, update);
    syncing = true;
    if (startEl) startEl.value = startYmd;
    if (endEl) endEl.value = endYmd;
    syncing = false;
    if (!fromUrl) writeUrlDates(startYmd, endYmd);
    schedule();
  }

  function onInputsChange() {
    if (syncing) return;
    var s = startEl && startEl.value;
    var e = endEl && endEl.value;
    if (!s || !e) return;
    applyDateWindow(s, e, false);
  }

  function injectToolbar() {
    if (document.getElementById('adaptive-date-window')) return;
    var bounds = dataXBoundsYmd() || ['', ''];
    var current = visibleYmd() || bounds;
    var url = readUrlDates();
    if (url[0] && url[1]) current = url;

    var bar = document.createElement('div');
    bar.id = 'adaptive-date-window';
    bar.style.cssText =
      'position:sticky;top:0;z-index:30;display:flex;flex-wrap:wrap;gap:10px 16px;' +
      'align-items:center;padding:8px 12px;background:#f6f7f8;border-bottom:1px solid #d0d5dd;' +
      'font:13px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#1f2a37;';

    function labeledInput(id, label, value) {
      var wrap = document.createElement('label');
      wrap.style.cssText = 'display:flex;align-items:center;gap:6px;';
      wrap.appendChild(document.createTextNode(label));
      var inp = document.createElement('input');
      inp.type = 'date';
      inp.id = id;
      inp.value = value || '';
      inp.min = bounds[0] || '';
      inp.max = bounds[1] || '';
      inp.style.cssText = 'padding:2px 6px;font:13px sans-serif;';
      wrap.appendChild(inp);
      bar.appendChild(wrap);
      return inp;
    }

    startEl = labeledInput('plot-start-date', 'start-date', current[0]);
    endEl = labeledInput('plot-end-date', 'end-date', current[1]);
    startEl.addEventListener('change', onInputsChange);
    endEl.addEventListener('change', onInputsChange);

    var applyBtn = document.createElement('button');
    applyBtn.type = 'button';
    applyBtn.textContent = '应用';
    applyBtn.style.cssText = 'padding:3px 10px;cursor:pointer;';
    applyBtn.addEventListener('click', onInputsChange);
    bar.appendChild(applyBtn);

    var resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.textContent = '全区间';
    resetBtn.style.cssText = 'padding:3px 10px;cursor:pointer;';
    resetBtn.addEventListener('click', function () {
      var b = dataXBoundsYmd();
      if (!b) return;
      applyDateWindow(b[0], b[1], false);
    });
    bar.appendChild(resetBtn);

    var hint = document.createElement('span');
    hint.style.cssText = 'color:#667085;font-size:12px;';
    hint.textContent =
      '改日期后各行 Y 轴按窗口重算，红绿三角等标注会跟着挂到窗口轨道；框选 X 也会同步日期。';
    bar.appendChild(hint);

    var parent = plot.parentNode || document.body;
    parent.insertBefore(bar, plot);

    if (url[0] && url[1]) {
      applyDateWindow(url[0], url[1], true);
    }
  }

  plot.on('plotly_relayout', function (ev) {
    if (!ev) return;
    var keys = Object.keys(ev);
    var xChanged = keys.some(function (k) {
      return k.indexOf('xaxis.range') === 0 || k === 'xaxis.autorange';
    });
    if (!xChanged) return;
    syncInputsFromPlot();
    schedule();
  });

  injectToolbar();
})();
"""


def write_adaptive_html(fig, path: str | Path, *, include_plotlyjs: str = "cdn") -> None:
    """Write Plotly figure HTML with date-window toolbar and X-zoom Y-autoscale."""
    fig.write_html(
        str(Path(path)),
        include_plotlyjs=include_plotlyjs,
        post_script=Y_AUTOSCALE_ON_X_ZOOM_JS,
        config={"scrollZoom": False, "displaylogo": False},
    )


__all__ = ["Y_AUTOSCALE_ON_X_ZOOM_JS", "write_adaptive_html"]
