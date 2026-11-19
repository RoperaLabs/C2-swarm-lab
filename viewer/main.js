// Static viewer for c2lab-world/1 files. Conventions: see .ia/grafo/mundo.md.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const TYPE_COLORS = { box: 0x8899aa, cylinder: 0xaa9977, sphere: 0x77aa88, capsule: 0xaa7788, cone: 0x88aa66, torus: 0xcc8844 };
const TAG_COLORS = { pillar: 0x8899aa, ring: 0xe0a030, arch: 0xd06040, slab: 0x7090c0, wall: 0x998877, sphere: 0x70b090,
  building: 0x8a8fa0, pole: 0x666666, cable: 0x222222, trunk: 0x7a5a3a, crown: 0x4a9a4a, bush: 0x6aaa4a, log: 0x6a4a2a,
  bridge: 0xb0a080, ceiling: 0x999999, column: 0xa0a0a0, rack: 0x6080a0, beam: 0x808080, lamp: 0xf0e080 };

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1e24);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(1, -2, 3);
scene.add(sun);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});

// Unit geometries in the local frame (axis = z, as in the spec).
const G = {
  box: new THREE.BoxGeometry(2, 2, 2),
  cylinder: new THREE.CylinderGeometry(1, 1, 2, 16).rotateX(Math.PI / 2),          // y axis -> z
  sphere: new THREE.SphereGeometry(1, 16, 12),
  cone: new THREE.ConeGeometry(1, 1, 16).translate(0, 0.5, 0).rotateX(Math.PI / 2), // base at origin, apex +z
  cyl1: new THREE.CylinderGeometry(1, 1, 1, 12).rotateX(Math.PI / 2),              // capsule body, length 1
};
const mats = new Map();
const mat = (c) => { if (!mats.has(c)) mats.set(c, new THREE.MeshLambertMaterial({ color: c })); return mats.get(c); };

const V = (a) => new THREE.Vector3(...a);
const Q = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // wxyz -> three xyzw
const M = (c, q, s) => new THREE.Matrix4().compose(V(c), q, s);
const S = (x, y, z) => new THREE.Vector3(x, y, z);

let content = null, world = null;
const hidden = new Set();

// Instanced pieces of one primitive (torus is handled separately).
function pieces(p) {
  const color = TAG_COLORS[p.tag] ?? TYPE_COLORS[p.type] ?? 0x999999;
  const id = new THREE.Quaternion();
  switch (p.type) {
    case 'box': return [{ k: 'box', color, m: M(p.c, Q(p.q), V(p.h)) }];
    case 'cylinder': return [{ k: 'cylinder', color, m: M(p.c, Q(p.q), S(p.r, p.r, p.hh)) }];
    case 'sphere': return [{ k: 'sphere', color, m: M(p.c, id, S(p.r, p.r, p.r)) }];
    case 'cone': return [{ k: 'cone', color, m: M(p.c, Q(p.q), S(p.r, p.r, p.h)) }];
    case 'capsule': {
      const a = V(p.a), b = V(p.b), d = b.clone().sub(a), len = d.length();
      const q = new THREE.Quaternion();
      if (len > 1e-9) q.setFromUnitVectors(S(0, 0, 1), d.divideScalar(len));
      return [
        { k: 'cyl1', color, m: M(a.clone().add(b).multiplyScalar(0.5).toArray(), q, S(p.r, p.r, len)) },
        { k: 'sphere', color, m: M(p.a, id, S(p.r, p.r, p.r)) },
        { k: 'sphere', color, m: M(p.b, id, S(p.r, p.r, p.r)) },
      ];
    }
  }
  return [];
}

function build() {
  if (content) { scene.remove(content); content.traverse(o => { if (o.isMesh && !o.isInstancedMesh) o.geometry.dispose(); }); }
  content = new THREE.Group();
  scene.add(content);
  const groups = new Map();
  for (const p of world.primitives) {
    if (hidden.has(p.tag)) continue;
    if (p.type === 'torus') {
      // three's arc starts at +x; the spec arc is centred on +y with length 2*arc.
      // ponytail: arc ends are flat, not rounded as in the SDF.
      const mesh = new THREE.Mesh(new THREE.TorusGeometry(p.R, p.r, 12, 48, 2 * p.arc), mat(TAG_COLORS[p.tag] ?? TYPE_COLORS.torus));
      mesh.quaternion.copy(Q(p.q)).multiply(new THREE.Quaternion().setFromAxisAngle(S(0, 0, 1), Math.PI / 2 - p.arc));
      mesh.position.copy(V(p.c));
      content.add(mesh);
      continue;
    }
    for (const i of pieces(p)) {
      const key = i.k + i.color;
      if (!groups.has(key)) groups.set(key, { k: i.k, color: i.color, ms: [] });
      groups.get(key).ms.push(i.m);
    }
  }
  for (const g of groups.values()) {
    const im = new THREE.InstancedMesh(G[g.k], mat(g.color), g.ms.length);
    g.ms.forEach((m, n) => im.setMatrixAt(n, m));
    content.add(im);
  }
}

function label(text, pos) {
  const cv = document.createElement('canvas');
  cv.width = 64; cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.font = 'bold 48px monospace'; ctx.textAlign = 'center'; ctx.fillText(text, 32, 50);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false }));
  s.position.copy(pos); s.scale.setScalar(4);
  return s;
}

let statics = null;
function load(w) {
  world = w;
  hidden.clear();
  if (statics) { scene.remove(statics); statics.traverse(o => o.geometry?.dispose()); }
  statics = new THREE.Group();
  scene.add(statics);
  const [lo, hi] = w.bounds.map(V);
  const size = hi.clone().sub(lo), ctr = lo.clone().add(hi).multiplyScalar(0.5);

  // Terrain: nx x ny nodes covering bounds in xy; heights[i][j] at x = i, y = j.
  const H = w.terrain.heights, nx = H.length, ny = H[0].length;
  const pos = [], idx = [];
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++)
    pos.push(lo.x + i * size.x / (nx - 1), lo.y + j * size.y / (ny - 1), H[i][j]);
  for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny - 1; j++) {
    const a = i * ny + j, b = a + ny, c = a + 1, d = b + 1;
    idx.push(a, b, c, b, d, c);
  }
  const tg = new THREE.BufferGeometry();
  tg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  tg.setIndex(idx); tg.computeVertexNormals();
  statics.add(new THREE.Mesh(tg, new THREE.MeshLambertMaterial({ color: 0x55664a, side: THREE.DoubleSide })));

  // Geofence: translucent box with edges.
  const bg = new THREE.BoxGeometry(size.x, size.y, size.z).translate(ctr.x, ctr.y, ctr.z);
  statics.add(new THREE.Mesh(bg, new THREE.MeshBasicMaterial({ color: 0x66aaff, transparent: true, opacity: 0.06, depthWrite: false })));
  statics.add(new THREE.LineSegments(new THREE.EdgesGeometry(bg), new THREE.LineBasicMaterial({ color: 0x66aaff })));

  // Anchors.
  for (const [name, a] of Object.entries(w.anchors)) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.6, 12, 8), new THREE.MeshBasicMaterial({ color: name === 'A' ? 0x40ff60 : 0xff5050 }));
    m.position.copy(V(a));
    statics.add(m, label(name, V(a).add(S(0, 0, 3))));
  }

  const r = size.length();
  camera.position.set(ctr.x - 0.5 * r, ctr.y - 0.8 * r, ctr.z + 0.6 * r);
  controls.target.copy(ctr);
  camera.far = r * 10; camera.updateProjectionMatrix();
  controls.update();
  build();
  panel();
}

const panelEl = document.getElementById('panel');
const fpsEl = document.createElement('div');
function panel() {
  const counts = {}, tagCount = {};
  for (const p of world.primitives) { counts[p.type] = (counts[p.type] || 0) + 1; tagCount[p.tag] = (tagCount[p.tag] || 0) + 1; }
  panelEl.replaceChildren();
  const info = document.createElement('div');
  info.style.whiteSpace = 'pre-wrap';
  info.textContent = `biome: ${world.biome}  seed: ${world.seed}  prims: ${world.primitives.length}\n` +
    Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join('  ');
  const file = document.createElement('input');
  file.type = 'file'; file.accept = '.json';
  file.onchange = () => file.files[0] && file.files[0].text().then(t => load(JSON.parse(t)));
  panelEl.append(fpsEl, info, file);
  for (const [tag, n] of Object.entries(tagCount)) {
    const l = document.createElement('label'), cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = true;
    cb.onchange = () => { cb.checked ? hidden.delete(tag) : hidden.add(tag); build(); };
    l.append(cb, ` ${tag} (${n})`);
    panelEl.append(l);
  }
}

addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => { e.preventDefault(); e.dataTransfer.files[0]?.text().then(t => load(JSON.parse(t))); });

// --- replay of a run (python -m c2lab run ... writes replay.json) ----------------------------------------------
const TEAM_COLORS = [0x4aa8ff, 0xff6a4a, 0x7ad35a, 0xe0c040];
let run = null, play = true, speed = 1, tNow = 0;
function loadRun(r) {
  r.links = r.links || []; r.covered = r.covered || [];
  load(r.world);
  if (run) scene.remove(run.group);
  const n = r.teams.length, teams = [...new Set(r.teams)];
  const group = new THREE.Group();
  const cone = new THREE.ConeGeometry(0.35, 1.0, 10).rotateZ(-Math.PI / 2);   // apex along body +x
  const mesh = new THREE.InstancedMesh(cone, new THREE.MeshLambertMaterial(), n);
  r.teams.forEach((t, i) => mesh.setColorAt(i, new THREE.Color(TEAM_COLORS[teams.indexOf(t) % TEAM_COLORS.length])));
  group.add(mesh);
  const trails = [];
  for (let i = 0; i < n; i++) {
    const line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: TEAM_COLORS[teams.indexOf(r.teams[i])], transparent: true, opacity: 0.5 }));
    trails.push(line);
    group.add(line);
  }
  fillTrails(r, trails);
  // radio links (one segment per delivered packet pair, refreshed from the latest recorded slot)
  const maxPairs = (n + 1) * n;
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxPairs * 6), 3));
  const links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: 0xffa040, transparent: true, opacity: 0.55 }));
  group.add(links);
  if (r.station) {
    const gs = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    gs.position.copy(V(r.station));
    group.add(gs, label('GS', V(r.station).add(S(0, 0, 2))));
  }
  if (r.covered.length) {           // surface voxels the swarm's sensors touched (coverage mission)
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(r.covered.flat(), 3));
    group.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x40ffb0, size: 0.6 })));
  }
  scene.add(group);
  run = { r, mesh, trails, group, links, follow: -1 };
  tNow = 0;
  const bar = document.createElement('div');
  const btn = document.createElement('button'), slider = document.createElement('input'), sp = document.createElement('select');
  const lbl = document.createElement('span');
  btn.textContent = 'pause'; btn.onclick = () => { play = !play; btn.textContent = play ? 'pause' : 'play'; };
  slider.type = 'range'; slider.min = 0; slider.max = r.t.length - 1; slider.value = 0; slider.style.width = '100%';
  slider.oninput = () => { tNow = r.t[+slider.value]; run.edge = +slider.value >= r.t.length - 2; };
  for (const s of [0.25, 0.5, 1, 2, 4, 8]) { const o = document.createElement('option'); o.value = s; o.textContent = `${s}x`; if (s === 1) o.selected = true; sp.append(o); }
  sp.onchange = () => { speed = +sp.value; };
  const follow = document.createElement('select');
  follow.append(new Option('free camera', -1), ...r.teams.map((t, i) => new Option(`follow ${i} (${t})`, i)));
  follow.onchange = () => { run.follow = +follow.value; run.prev = null; };
  const showLinks = document.createElement('label'), cbl = document.createElement('input');
  cbl.type = 'checkbox'; cbl.checked = true; cbl.onchange = () => { links.visible = cbl.checked; };
  showLinks.append(cbl, ' radio links');
  bar.append(btn, sp, lbl, slider, follow, showLinks);
  panelEl.append(bar);
  const evs = document.createElement('div');      // events: click to jump there
  evs.style.cssText = 'max-height:140px;overflow:auto;font-size:11px;margin-top:4px';
  const info = document.createElement('div');
  info.style.whiteSpace = 'pre-wrap';
  panelEl.append(evs, info);
  run.ui = { slider, lbl, evs, info };
  fillPanel(r);
}

function fillTrails(r, trails) {
  trails.forEach((line, i) => {
    const pos = new Float32Array(r.t.length * 3);
    r.p.forEach((f, k) => pos.set(f.slice(3 * i, 3 * i + 3), 3 * k));
    line.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    line.geometry.computeBoundingSphere();
  });
}

function fillPanel(r) {
  const { evs, info } = run.ui;
  evs.replaceChildren(...r.events.filter(e => e.kind !== 'takeoff').map(e => {
    const b = document.createElement('div');
    b.textContent = `${e.t.toFixed(1)} s  #${e.agent}  ${e.kind}`;
    b.style.cursor = 'pointer';
    b.onclick = () => { tNow = e.t; run.edge = false; };
    return b;
  }));
  const m = r.metrics, pct = (x) => `${(100 * x).toFixed(0)} %`;
  const score = m.live ? `LIVE ${m.seconds.toFixed(0)} s` : { a_to_b: `arrival ${pct(m.arrival_rate)}`,
    coverage: `coverage ${pct(m.coverage)}`, pursuit: `captured ${pct(m.capture_rate)}` }[m.mission] ?? '';
  info.textContent = `${m.mission}: ${score}  crashes ${m.crashes}  agents ${m.agents}` + (m.privileged ? '  (privileged)' : '');
}

function updateRun(d) {          // live mode: the run grew; keep the camera and, at the live edge, stay on it
  const r = run.r;
  Object.assign(r, { t: d.t, p: d.p, q: d.q, alive: d.alive, links: d.links || [], events: d.events, metrics: d.metrics });
  fillTrails(r, run.trails);
  run.ui.slider.max = r.t.length - 1;
  fillPanel(r);
  if (run.edge) tNow = r.t[r.t.length - 1];
}

function showFrame() {
  const { r, mesh, trails, ui } = run, n = r.teams.length;
  let k = r.t.findIndex(t => t > tNow);
  k = k < 0 ? r.t.length - 1 : Math.max(k - 1, 0);
  const m = new THREE.Matrix4(), one = new THREE.Vector3(1, 1, 1), zero = new THREE.Vector3(0, 0, 0);
  for (let i = 0; i < n; i++) {
    const p = r.p[k].slice(3 * i, 3 * i + 3), q = r.q[k].slice(4 * i, 4 * i + 4);
    m.compose(V(p), Q(q), r.alive[k][i] ? one : zero);
    mesh.setMatrixAt(i, m);
    trails[i].geometry.setDrawRange(0, k + 1);
  }
  mesh.instanceMatrix.needsUpdate = true;
  const pos = (i) => (i < n ? r.p[k].slice(3 * i, 3 * i + 3) : r.station);
  let li = r.links.findIndex(l => l[0] > tNow);
  li = li < 0 ? r.links.length - 1 : li - 1;
  const arr = run.links.geometry.attributes.position.array;
  const pairs = li >= 0 ? r.links[li][1].filter(([a, b]) => pos(a) && pos(b)) : [];
  pairs.forEach(([a, b], j) => { arr.set(pos(a), 6 * j); arr.set(pos(b), 6 * j + 3); });
  run.links.geometry.setDrawRange(0, 2 * pairs.length);
  run.links.geometry.attributes.position.needsUpdate = true;
  if (run.follow >= 0) {           // keep the camera's offset to the followed drone
    const target = V(pos(run.follow));
    if (run.prev) camera.position.add(target.clone().sub(run.prev));
    controls.target.copy(target);
    run.prev = target;
  }
  ui.slider.value = k;
  ui.lbl.textContent = ` t = ${r.t[k].toFixed(1)} s `;
}

let frames = 0, t0 = performance.now(), last = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  if (run) {
    if (play) tNow = Math.min(tNow + speed * (now - last) / 1000, run.r.t[run.r.t.length - 1]);
    showFrame();
  }
  last = now;
  controls.update();
  renderer.render(scene, camera);
  frames++;
  if (now - t0 >= 500) { fpsEl.textContent = `fps: ${(frames * 1000 / (now - t0)).toFixed(0)}`; frames = 0; t0 = now; }
});

const params = new URLSearchParams(location.search);
const sweepDir = params.get('sweep');
let src = params.get('run') || params.get('world') || 'sample_world.json', shown = null;
const fresh = (u) => fetch(`${u}${u.includes('?') ? '&' : '?'}ts=${Date.now()}`).then(r => r.json());
const poll = () => {             // live: poll the replay while the run writes it (run --live, sweep --live)
  const u = src;
  fresh(u).then(d => {
    if (u !== src) return;
    if (shown !== u) { loadRun(d); run.edge = true; tNow = d.t[d.t.length - 1]; shown = u; }
    else if (d.t.length !== run.r.t.length || !d.metrics.live) updateRun(d);
  }).catch(() => { if (!run) panelEl.textContent = 'waiting for the run...'; })
    .finally(() => { if (sweepDir || !run || run.r.metrics.live) setTimeout(poll, 2000); });
};
if (sweepDir) {                  // sweep --live: follow what one worker process is running, run after run
  const box = document.createElement('div'), sel = document.createElement('select');
  box.style.cssText = 'position:fixed;top:8px;right:8px;background:#000c;color:#ddd;padding:6px;font:12px monospace';
  box.append('worker ', sel);
  document.body.append(box);
  let slots = 0, follow = 0;
  sel.onchange = () => { follow = +sel.value; };
  const workers = () => Promise.all([...Array(slots + 1).keys()].map(k => fresh(`${sweepDir}/_live/${k}.json`).catch(() => null)))
    .then(ws => {                // probe one slot past the last found: a late worker joins the list
      slots = Math.max(slots, ws.findLastIndex(Boolean) + 1);
      sel.replaceChildren(...ws.map((w, k) => w && new Option(`${k}: ${w.run}`, k)).filter(Boolean));
      sel.value = follow;
      if (ws[follow]) src = `${sweepDir}/${ws[follow].run}/replay.json`;
    })
    .finally(() => setTimeout(workers, 2000));
  workers().then(poll);
} else if (params.get('live')) {
  poll();
} else {
  fetch(src).then(r => r.json()).then(d => (d.frames || d.t ? loadRun(d) : load(d)))
    .catch(e => { panelEl.textContent = `load error: ${e}`; console.error(e); });
}
