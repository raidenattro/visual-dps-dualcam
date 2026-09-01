import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { meshCells, makeLayerMesh, DEFAULT_LAYER_PITCH } from '../lib/dualcamGeom.js';
import { AISLE_3D_EDGES, MAX_BONE_M } from '../lib/cocoSkeleton.js';

/** 与 pick-state dualcam_player 一致：墙1 绿、墙2 蓝（不是标注页的橙/青） */
const WALL_FILL = { 1: 0x3d8a5a, 2: 0x2f6f9f };
const WALL_LINE = { 1: 0x55cc88, 2: 0x66aaff };
const CAM_COLOR = { L: 0xffcc66, R: 0xff8866 };
const PERSON_HEX = [0x66e0ff, 0xffcc66, 0xc9a0ff, 0x5cf08a, 0xff8866, 0x8ab4ff, 0xf0a0c0, 0xd0d060];
const MAX_PERSONS = PERSON_HEX.length;
const WRIST_NEAR = 0x3dd68c;
const WRIST_FAR = 0xffa23c;

/** 按巷道记住轨道视角，轮询刷新 aisle 对象时不要弹回默认角 */
const VIEW_BY_AISLE = new Map();

function flatCorners(c) {
  return [...c[0], ...c[1], ...c[2], ...c[0], ...c[2], ...c[3]];
}

function edgeCorners(c) {
  return [...c[0], ...c[1], ...c[1], ...c[2], ...c[2], ...c[3], ...c[3], ...c[0]];
}

function slotMeshesOf(aisle) {
  const meshes = aisle?.slot_meshes || [];
  if (meshes.length) return meshes;
  const walls = aisle?.solved?.walls || [];
  return walls.map((w) => makeLayerMesh(w.wall_id, w.corners, DEFAULT_LAYER_PITCH, 4, 4));
}

function aisleGeomKey(aisle) {
  if (!aisle) return '';
  const meshes = aisle.slot_meshes || [];
  const walls = aisle.solved?.walls || [];
  const meshSig = meshes.map((m) => `${m.wall_id}:${m.rows}:${m.cols}`).join(',');
  const wallSig = walls.map((w) => w.wall_id).join(',');
  return `${aisle.aisle_id}|${wallSig}|${meshSig}|${meshes.length}`;
}

function slotTokens(mesh, cell) {
  const shelf = String(mesh.shelf_code || '').trim();
  const box = String(cell.box_id || cell.slot_key || '');
  const wid = Number(mesh.wall_id);
  const toks = [];
  if (shelf && box) toks.push(`${shelf}:${box}`);
  if (box) {
    toks.push(`Box_${box}`);
    toks.push(`${wid}:${box}`);
    toks.push(`wall${wid}:${box}`);
    toks.push(`w${wid}-${box}`);
  }
  return toks;
}

function alarmOn(alarm, idx) {
  if (!alarm || typeof alarm !== 'object') return false;
  return Boolean(alarm[idx] || alarm[String(idx)]);
}

function isPreviewPerson(person) {
  return person?.preview === true;
}

function makeSkelRig(scene) {
  const group = new THREE.Group();
  scene.add(group);
  const people = [];
  const jointGeo = new THREE.SphereGeometry(0.032, 8, 8);
  const wristGeo = new THREE.SphereGeometry(0.048, 10, 10);
  for (let p = 0; p < MAX_PERSONS; p += 1) {
    const g = new THREE.Group();
    const jointMat = new THREE.MeshBasicMaterial({ color: PERSON_HEX[p] });
    const boneMat = new THREE.LineBasicMaterial({ color: PERSON_HEX[p] });
    const joints = [];
    for (let i = 0; i < 17; i += 1) {
      const m = new THREE.Mesh(i === 9 || i === 10 ? wristGeo : jointGeo, jointMat);
      m.visible = false;
      g.add(m);
      joints.push(m);
    }
    const pos = new Float32Array(AISLE_3D_EDGES.length * 2 * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setDrawRange(0, 0);
    g.add(new THREE.LineSegments(geo, boneMat));
    g.visible = false;
    group.add(g);
    people.push({ g, joints, pos, geo, jointMat, boneMat });
  }
  const wristNear = new THREE.MeshBasicMaterial({ color: WRIST_NEAR });
  const wristFar = new THREE.MeshBasicMaterial({ color: WRIST_FAR });
  return { people, wristNear, wristFar };
}

function dist3(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function torsoCenter(xyz) {
  const pts = [5, 6, 11, 12].map((i) => xyz?.[i]).filter((p) => p && p.length >= 3);
  if (!pts.length) return null;
  const s = pts.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0]);
  return [s[0] / pts.length, s[1] / pts.length, s[2] / pts.length];
}

function updateSkelRig(rig, people) {
  for (let p = 0; p < MAX_PERSONS; p += 1) {
    const slot = rig.people[p];
    const person = people?.[p];
    const xyz = person?.xyz;
    if (!xyz) {
      slot.g.visible = false;
      slot.geo.setDrawRange(0, 0);
      continue;
    }
    const preview = isPreviewPerson(person);
    const mid = preview ? torsoCenter(xyz) : null;
    slot.g.visible = true;
    const alarm = person?.wrist_alarm || {};
    const srcs = person?.src || [];
    let n = 0;
    for (let i = 0; i < 17; i += 1) {
      const m = slot.joints[i];
      const pt = xyz[i];
      const src = srcs[i];
      const glued = preview && (src === 'Lmono' || src === 'Rmono' || src === 'Lhold' || src === 'Rhold');
      const far = preview && mid && pt && pt.length >= 3 && dist3(pt, mid) > 1.35;
      if (!pt || pt.length < 3 || far || (glued && i <= 4)) {
        m.visible = false;
        continue;
      }
      m.visible = true;
      m.position.set(pt[0], pt[1], pt[2]);
      if (i === 9 || i === 10) {
        m.material = alarmOn(alarm, i) ? rig.wristNear : rig.wristFar;
      } else {
        m.material = slot.jointMat;
      }
    }
    AISLE_3D_EDGES.forEach(([a, b]) => {
      const pa = xyz[a];
      const pb = xyz[b];
      if (!pa || !pb || pa.length < 3 || pb.length < 3) return;
      // 只过滤单路贴墙预览的超长骨，立体骨架按实验仓播放器原样画
      if (preview && dist3(pa, pb) > MAX_BONE_M) return;
      if (preview && mid && (dist3(pa, mid) > 1.35 || dist3(pb, mid) > 1.35)) return;
      slot.pos[n] = pa[0]; slot.pos[n + 1] = pa[1]; slot.pos[n + 2] = pa[2];
      slot.pos[n + 3] = pb[0]; slot.pos[n + 4] = pb[1]; slot.pos[n + 5] = pb[2];
      n += 6;
    });
    slot.geo.attributes.position.needsUpdate = true;
    slot.geo.setDrawRange(0, n / 3);
  }
}

function paintSlots(slotObjs, collisions, alarms) {
  const keys = new Set([...(collisions || []), ...(alarms || [])].map(String));
  slotObjs.forEach((s) => {
    const on = s.tokens.some((t) => keys.has(t));
    s.fill.material = on ? s.alarmFill : s.idleFill;
    s.line.material = on ? s.alarmLine : s.idleLine;
  });
}

function buildStatic(scene, aisle) {
  const group = new THREE.Group();
  group.name = 'aisle-static';
  const solved = aisle?.solved?.ok ? aisle.solved : null;
  const walls = solved?.walls || [];
  walls.forEach((w) => {
    const c = w.corners;
    if (!c || c.length < 4) return;
    const color = WALL_FILL[Number(w.wall_id)] || 0x888888;
    const lineColor = WALL_LINE[Number(w.wall_id)] || 0x888888;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(flatCorners(c), 3));
    group.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
      color,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthWrite: false,
    })));
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(edgeCorners(c), 3));
    group.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: lineColor })));
  });

  const idleFill = new THREE.MeshBasicMaterial({
    color: 0x3a6a88, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false,
  });
  const idleLine = new THREE.LineBasicMaterial({ color: 0x7ec8ff });
  const alarmFill = new THREE.MeshBasicMaterial({
    color: 0xff3344, transparent: true, opacity: 0.78, side: THREE.DoubleSide, depthWrite: false,
  });
  const alarmLine = new THREE.LineBasicMaterial({ color: 0xff5566 });
  const slotObjs = [];
  for (const mesh of slotMeshesOf(aisle)) {
    for (const cell of meshCells(mesh)) {
      const c = cell.corners;
      if (!c || c.length < 4) continue;
      const fill = new THREE.Mesh(
        new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(flatCorners(c), 3)),
        idleFill,
      );
      const line = new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(edgeCorners(c), 3)),
        idleLine,
      );
      group.add(fill);
      group.add(line);
      slotObjs.push({
        tokens: slotTokens(mesh, cell),
        fill,
        line,
        idleFill,
        idleLine,
        alarmFill,
        alarmLine,
      });
    }
  }

  const cams = solved?.cameras || {};
  ['L', 'R'].forEach((role) => {
    const cam = cams[role];
    if (!cam?.C || !cam?.fwd) return;
    const color = CAM_COLOR[role];
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 10, 10),
      new THREE.MeshBasicMaterial({ color }),
    );
    ball.position.set(...cam.C);
    group.add(ball);
    group.add(new THREE.ArrowHelper(
      new THREE.Vector3(...cam.fwd),
      new THREE.Vector3(...cam.C),
      0.55,
      color,
      0.12,
      0.08,
    ));
  });
  scene.add(group);
  return { group, slotObjs };
}

/** 巷道 3D：墙/货格静态，人骨架与碰撞按 overlay 刷新；轨道视角按巷道记住。 */
export default function AisleScene3D({ aisle, persons3d = [], collisions = [], alarms = [] }) {
  const canvasRef = useRef(null);
  const liveRef = useRef({ people: persons3d, collisions, alarms, aisle });
  liveRef.current = { people: persons3d || [], collisions: collisions || [], alarms: alarms || [], aisle };
  const geomKey = useMemo(() => aisleGeomKey(aisle), [aisle]);
  const aisleId = aisle?.aisle_id || '';

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(1);
    renderer.setClearColor(0x111111, 1);
    const scene = new THREE.Scene();
    const viewCam = new THREE.PerspectiveCamera(50, 1, 0.05, 40);
    const saved = VIEW_BY_AISLE.get(aisleId);
    if (saved?.pos && saved?.target) {
      viewCam.position.fromArray(saved.pos);
    } else {
      viewCam.position.set(3.6, 3.4, 1.1);
    }
    const controls = new OrbitControls(viewCam, canvas);
    if (saved?.target) controls.target.fromArray(saved.target);
    else controls.target.set(0, 0.9, 1.1);
    controls.enableDamping = true;
    const persistView = () => {
      VIEW_BY_AISLE.set(aisleId, {
        pos: viewCam.position.toArray(),
        target: controls.target.toArray(),
      });
    };
    controls.addEventListener('change', persistView);
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(2, 6, 1);
    scene.add(dir);
    scene.add(new THREE.GridHelper(8, 16, 0x3a3a3a, 0x2a2a2a));

    let staticPack = buildStatic(scene, liveRef.current.aisle);
    const skelRig = makeSkelRig(scene);

    const resize = () => {
      const parent = canvas.parentElement;
      const w = Math.max(2, parent?.clientWidth || canvas.clientWidth || 2);
      const h = Math.max(2, parent?.clientHeight || canvas.clientHeight || 2);
      renderer.setSize(w, h, false);
      viewCam.aspect = w / h;
      viewCam.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    let raf = 0;
    let lastPaint = '';
    const tick = () => {
      const live = liveRef.current;
      updateSkelRig(skelRig, live.people);
      const sig = `${(live.collisions || []).join('|')}|${(live.alarms || []).join('|')}`;
      if (sig !== lastPaint) {
        lastPaint = sig;
        paintSlots(staticPack.slotObjs, live.collisions, live.alarms);
      }
      controls.update();
      renderer.render(scene, viewCam);
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      persistView();
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.removeEventListener('change', persistView);
      controls.dispose();
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          mats.forEach((m) => m.dispose?.());
        }
      });
    };
  }, [aisleId, geomKey]);

  return <canvas ref={canvasRef} className="aisle-3d-canvas" />;
}
