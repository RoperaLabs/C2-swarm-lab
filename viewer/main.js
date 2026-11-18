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

let frames = 0, t0 = performance.now();
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
  frames++;
  const t = performance.now();
  if (t - t0 >= 500) { fpsEl.textContent = `fps: ${(frames * 1000 / (t - t0)).toFixed(0)}`; frames = 0; t0 = t; }
});

const url = new URLSearchParams(location.search).get('world') || 'sample_world.json';
fetch(url).then(r => r.json()).then(load).catch(e => { panelEl.textContent = `load error: ${e}`; console.error(e); });
