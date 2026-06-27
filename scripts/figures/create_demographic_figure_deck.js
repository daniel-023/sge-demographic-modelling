const path = require('path');
const pptxgen = require('/Users/daniel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Daniel Kainovan Handoyo';
pptx.company = 'NTU';
pptx.subject = 'Demographic prediction pipeline figures';
pptx.title = 'Demographic Prediction Pipeline Figures';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'DejaVu Sans',
  bodyFontFace: 'DejaVu Sans',
  lang: 'en-US',
};
pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'CUSTOM_WIDE';
pptx.margin = 0;

const OUT = path.resolve(__dirname, '../../figures/demographic_pipeline_figures.pptx');

const C = {
  text: '1F2937',
  muted: '6B7280',
  line: '4B5563',
  lightLine: 'B7C0CE',
  blueEdge: '6B7FAD',
  blueFill: 'E8EDF5',
  greenEdge: '4A8C6A',
  greenFill: 'DFF0E6',
  yellowEdge: 'C9A227',
  yellowFill: 'FEF9E7',
  purpleEdge: '8B6AAD',
  purpleFill: 'F0E8F7',
  purpleOuter: 'A88CC4',
  purpleOuterFill: 'FAF7FD',
  mlpBlueEdge: '2563EB',
  mlpBlueFill: 'DBEAFE',
  mlpGreenEdge: '16A34A',
  mlpGreenFill: 'DCFCE7',
  orangeEdge: 'D97706',
  orangeFill: 'FEF3C7',
  divider: 'D1D5DB',
};

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: opts.margin ?? 0,
    fontFace: opts.fontFace || 'DejaVu Sans',
    fontSize: opts.fontSize || 14,
    color: opts.color || C.text,
    bold: opts.bold || false,
    italic: opts.italic || false,
    align: opts.align || 'center',
    valign: opts.valign || 'mid',
    breakLine: false,
    paraSpaceAfterPt: 0,
  });
}

function roundedBox(slide, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: opts.radius || 0.08,
    fill: { color: opts.fill || 'FFFFFF', transparency: opts.transparency || 0 },
    line: { color: opts.stroke || C.line, width: opts.lineWidth || 1.2 },
  });
  if (opts.text) {
    addText(slide, opts.text, x + (opts.padX || 0.05), y + (opts.padY || 0.03),
      w - 2 * (opts.padX || 0.05), h - 2 * (opts.padY || 0.03), opts.textOpts || {});
  }
}

function arrow(slide, x1, y1, x2, y2, opts = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color: opts.color || C.line,
      width: opts.width || 1.35,
      beginArrowType: opts.beginArrowType || 'none',
      endArrowType: opts.endArrowType || 'triangle',
      dash: opts.dash || 'solid',
      transparency: opts.transparency || 0,
    },
  });
}

function line(slide, x1, y1, x2, y2, opts = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color: opts.color || C.line,
      width: opts.width || 1.0,
      dash: opts.dash || 'solid',
      transparency: opts.transparency || 0,
    },
  });
}

function pill(slide, x, y, w, h, text, stroke, fill) {
  roundedBox(slide, x, y, w, h, {
    stroke,
    fill,
    lineWidth: 1.2,
    text,
    textOpts: { fontSize: 14, color: C.text },
  });
}

function addMultiline(slide, lines, cx, cy, w, lineH = 0.26) {
  const totalH = lineH * lines.length;
  const startY = cy - totalH / 2;
  lines.forEach((l, idx) => {
    addText(slide, l.text, cx - w / 2, startY + idx * lineH, w, lineH, {
      fontSize: l.size,
      bold: l.bold || false,
      color: l.color || C.text,
    });
  });
}

function waveform(slide, x, y, w, h) {
  let seed = 42;
  function randn() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const u = (seed + 1) / 4294967297;
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const v = (seed + 1) / 4294967297;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const n = 95;
  let last = null;
  for (let i = 0; i < n; i += 1) {
    const t = i / (n - 1);
    const px = x + t * w;
    const envelope = Math.exp(-0.5 * Math.pow((t - 0.5) / 0.22, 2));
    const py = y + h / 2 + envelope * randn() * h * 0.18;
    if (last) line(slide, last.x, last.y, px, py, { color: C.text, width: 0.65 });
    last = { x: px, y: py };
  }
}

function embedding(slide, cx, cy, w, h, stroke = C.mlpBlueEdge, fill = C.mlpBlueFill) {
  slide.addShape(pptx.ShapeType.rect, {
    x: cx - w / 2, y: cy - h / 2, w, h,
    fill: { color: fill },
    line: { color: stroke, width: 1.2 },
  });
  for (let i = 1; i < 4; i += 1) {
    line(slide, cx - w / 2, cy - h / 2 + h * i / 4, cx + w / 2, cy - h / 2 + h * i / 4,
      { color: stroke, width: 0.9 });
  }
}

function addOverviewSlide() {
  const slide = pptx.addSlide();
  slide.background = { color: 'FFFFFF' };
  addText(slide, 'Overview of the demographic prediction pipeline', 0.75, 0.28, 11.85, 0.46, {
    fontFace: 'DejaVu Sans',
    fontSize: 22,
    bold: true,
  });

  pill(slide, 0.35, 1.05, 2.45, 0.43, '1. Input', C.blueEdge, C.blueFill);
  pill(slide, 3.55, 1.05, 2.45, 0.43, '2. Speech encoder', C.greenEdge, C.greenFill);
  pill(slide, 6.75, 1.05, 2.45, 0.43, '3. Classifier heads', C.yellowEdge, C.yellowFill);
  pill(slide, 9.95, 1.05, 2.45, 0.43, '4. Outputs', C.purpleEdge, C.purpleFill);

  roundedBox(slide, 0.30, 1.95, 2.5, 3.45, { stroke: C.blueEdge, fill: C.blueFill, lineWidth: 1.2 });
  addText(slide, 'National Speech\nCorpus', 0.55, 2.45, 2.0, 0.60, { fontSize: 13.5, bold: true });
  addText(slide, 'speech recordings', 0.55, 3.05, 2.0, 0.32, { fontSize: 14.5, bold: true });
  waveform(slide, 0.75, 3.47, 1.62, 1.05);

  roundedBox(slide, 3.55, 2.82, 2.45, 1.75, { stroke: C.greenEdge, fill: C.greenFill, lineWidth: 1.2 });
  addMultiline(slide, [
    { text: 'WavLM', size: 22, bold: true },
    { text: 'extracts frame-level', size: 14 },
    { text: 'representations', size: 14 },
  ], 4.775, 3.70, 2.1, 0.28);

  arrow(slide, 2.80, 3.70, 3.55, 3.70, { width: 1.3 });
  line(slide, 6.00, 3.70, 6.38, 3.70, { width: 1.2 });
  line(slide, 6.38, 2.78, 6.38, 4.64, { width: 1.2 });

  roundedBox(slide, 6.58, 2.36, 3.28, 1.58, { stroke: C.yellowEdge, fill: C.yellowFill, lineWidth: 1.2 });
  addMultiline(slide, [
    { text: 'Global-summary head', size: 14, bold: true },
    { text: '(MLP)', size: 14, bold: true },
    { text: 'Averages across the utterance', size: 13 },
    { text: 'before classification.', size: 13 },
  ], 8.22, 3.15, 2.80, 0.28);

  roundedBox(slide, 6.58, 4.23, 3.28, 1.58, { stroke: C.yellowEdge, fill: C.yellowFill, lineWidth: 1.2 });
  addMultiline(slide, [
    { text: 'Sequence-based head', size: 14, bold: true },
    { text: '(LSTM)', size: 14, bold: true },
    { text: 'Preserves temporal order', size: 13 },
    { text: 'before classification.', size: 13 },
  ], 8.22, 5.02, 2.80, 0.28);

  arrow(slide, 6.38, 2.78, 6.58, 2.95, { width: 1.2 });
  arrow(slide, 6.38, 4.64, 6.58, 5.02, { width: 1.2 });

  roundedBox(slide, 10.05, 1.83, 2.45, 4.15, {
    stroke: C.purpleOuter,
    fill: C.purpleOuterFill,
    lineWidth: 1.2,
  });
  addText(slide, 'Demographic attributes', 10.35, 2.30, 1.85, 0.36, { fontSize: 15, bold: true });
  roundedBox(slide, 10.40, 2.90, 1.75, 0.72, { stroke: C.purpleEdge, fill: C.purpleFill, lineWidth: 1.1, text: 'Age', textOpts: { fontSize: 18, bold: true } });
  roundedBox(slide, 10.40, 3.78, 1.75, 0.72, { stroke: C.purpleEdge, fill: C.purpleFill, lineWidth: 1.1, text: 'Gender', textOpts: { fontSize: 18, bold: true } });
  roundedBox(slide, 10.40, 4.66, 1.75, 0.72, { stroke: C.purpleEdge, fill: C.purpleFill, lineWidth: 1.1, text: 'Ethnicity', textOpts: { fontSize: 18, bold: true } });

  arrow(slide, 9.86, 3.14, 10.05, 3.20, { width: 1.25 });
  arrow(slide, 9.86, 5.02, 10.05, 4.90, { width: 1.25 });

  addText(slide, 'WavLM frame-level representations feed two classifier heads: a pooled MLP and a temporal LSTM.',
    2.0, 6.85, 9.3, 0.30, { fontSize: 13, italic: true, color: C.muted });
}

function addComparisonSlide() {
  const slide = pptx.addSlide();
  slide.background = { color: 'FFFFFF' };
  addText(slide, 'Comparison of the two modelling approaches', 0.55, 0.16, 12.25, 0.42, {
    fontFace: 'DejaVu Sans',
    fontSize: 22,
    bold: true,
  });
  line(slide, 6.665, 0.74, 6.665, 7.05, { color: C.divider, width: 1.0, dash: 'dash' });

  addText(slide, 'A.  Global-summary model (MLP)', 0.78, 0.78, 5.20, 0.34, {
    fontFace: 'DejaVu Sans',
    fontSize: 16,
    bold: true,
  });
  addText(slide, 'B.  Sequence-based model (LSTM)', 7.33, 0.78, 5.35, 0.34, {
    fontFace: 'DejaVu Sans',
    fontSize: 16,
    bold: true,
  });

  const ax = [1.25, 2.30, 3.35, 5.15];
  const ay = 6.05;
  ax.forEach((x) => embedding(slide, x, ay, 0.50, 1.02));
  addText(slide, 'h1', 1.08, 6.72, 0.34, 0.25, { fontSize: 14 });
  addText(slide, 'h2', 2.13, 6.72, 0.34, 0.25, { fontSize: 14 });
  addText(slide, 'h3', 3.18, 6.72, 0.34, 0.25, { fontSize: 14 });
  addText(slide, '...', 4.15, 6.13, 0.45, 0.25, { fontSize: 18, bold: true, color: C.muted });
  addText(slide, 'hT', 4.98, 6.72, 0.34, 0.25, { fontSize: 14 });

  roundedBox(slide, 2.10, 4.15, 2.45, 0.62, { stroke: C.mlpGreenEdge, fill: C.mlpGreenFill, lineWidth: 1.2, text: 'Mean Pooling', textOpts: { fontSize: 16 } });
  addText(slide, 'average all frames', 4.62, 4.28, 1.15, 0.26, { fontSize: 11, color: C.muted });
  ax.forEach((x) => arrow(slide, x, 5.54, 3.32, 4.77, { color: C.lightLine, width: 0.95 }));

  embedding(slide, 3.35, 3.45, 0.60, 0.72);
  addText(slide, 'h\u0304', 4.12, 3.34, 0.35, 0.35, { fontSize: 18 });
  arrow(slide, 3.35, 4.15, 3.35, 3.81, { color: C.lightLine, width: 1.0 });

  roundedBox(slide, 2.10, 2.43, 2.45, 0.62, { stroke: C.mlpGreenEdge, fill: C.mlpGreenFill, lineWidth: 1.2, text: 'Feedforward Layers', textOpts: { fontSize: 16 } });
  arrow(slide, 3.35, 3.09, 3.35, 3.05, { color: C.lightLine, width: 1.0 });

  roundedBox(slide, 2.10, 1.20, 2.45, 0.62, { stroke: C.orangeEdge, fill: C.orangeFill, lineWidth: 1.2, text: 'Prediction', textOpts: { fontSize: 17, bold: true } });
  arrow(slide, 3.35, 2.43, 3.35, 1.82, { color: C.lightLine, width: 1.0 });

  addText(slide, 'Captures global acoustic properties; discards temporal structure.',
    1.05, 7.03, 4.70, 0.28, { fontSize: 12.2, color: C.muted });

  const bx = [7.95, 9.00, 10.05, 11.85];
  const by = 6.05;
  bx.forEach((x) => embedding(slide, x, by, 0.50, 1.02));
  addText(slide, 'h1', 7.78, 6.72, 0.34, 0.25, { fontSize: 14 });
  addText(slide, 'h2', 8.83, 6.72, 0.34, 0.25, { fontSize: 14 });
  addText(slide, 'h3', 9.88, 6.72, 0.34, 0.25, { fontSize: 14 });
  addText(slide, '...', 10.78, 6.13, 0.45, 0.25, { fontSize: 18, bold: true, color: C.muted });
  addText(slide, 'hT', 11.68, 6.72, 0.34, 0.25, { fontSize: 14 });

  bx.forEach((x) => roundedBox(slide, x - 0.56, 4.20, 1.12, 0.56, { stroke: C.mlpGreenEdge, fill: C.mlpGreenFill, lineWidth: 1.1, text: 'LSTM', textOpts: { fontSize: 13 } }));
  bx.forEach((x) => arrow(slide, x, 5.54, x, 4.76, { color: C.lightLine, width: 1.0 }));
  addText(slide, '...', 10.85, 4.36, 0.45, 0.25, { fontSize: 18, bold: true, color: C.muted });

  line(slide, 7.95, 3.95, 11.85, 3.95, { color: C.lightLine, width: 0.9 });
  bx.forEach((x) => line(slide, x, 4.20, x, 3.95, { color: C.lightLine, width: 0.9 }));
  roundedBox(slide, 8.40, 3.18, 3.05, 0.62, { stroke: C.mlpGreenEdge, fill: C.mlpGreenFill, lineWidth: 1.2, text: 'Aggregated Hidden States', textOpts: { fontSize: 14 } });
  arrow(slide, 10.05, 3.95, 10.05, 3.80, { color: C.lightLine, width: 0.9 });

  roundedBox(slide, 8.40, 2.43, 3.05, 0.62, { stroke: C.mlpGreenEdge, fill: C.mlpGreenFill, lineWidth: 1.2, text: 'Classification Layer', textOpts: { fontSize: 16 } });
  arrow(slide, 10.05, 3.18, 10.05, 3.05, { color: C.lightLine, width: 1.0 });

  roundedBox(slide, 8.65, 1.20, 2.55, 0.62, { stroke: C.orangeEdge, fill: C.orangeFill, lineWidth: 1.2, text: 'Prediction', textOpts: { fontSize: 17, bold: true } });
  arrow(slide, 10.05, 2.43, 10.05, 1.82, { color: C.lightLine, width: 1.0 });

  addText(slide, 'Preserves time-varying rhythmic and intonational information.',
    7.85, 7.03, 4.65, 0.28, { fontSize: 12.2, color: C.muted });
}

addOverviewSlide();
addComparisonSlide();

pptx.writeFile({ fileName: OUT });
console.log(`Saved: ${OUT}`);
