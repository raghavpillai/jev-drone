import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Furnishings } from './furnishings.js';
import { TargetEffects } from './target-effects.js';

function woodTexture() {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 512;
  const c = canvas.getContext('2d');
  let seed = 18;
  const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let board = 0; board < 8; board++) {
    const light = 63 + random() * 4;
    c.fillStyle = `hsl(31, 24%, ${light}%)`; c.fillRect(board * 64, 0, 64, 512);
    for (let grain = 0; grain < 50; grain++) {
      const x = board * 64 + random() * 63;
      c.strokeStyle = `rgba(67,40,20,${random() * .12})`; c.lineWidth = .4 + random();
      c.beginPath(); c.moveTo(x, 0); c.bezierCurveTo(x + random() * 8, 170, x - random() * 8, 300, x, 512); c.stroke();
    }
    c.fillStyle = '#4f3a2428'; c.fillRect(board * 64, 0, 1, 512);
    c.fillRect(board * 64, board % 2 ? 190 : 400, 64, 1);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.repeat.set(5, 3);
  return texture;
}

export class HouseView {
  constructor(container) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.15;
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#b9c1b9');
    this.camera = new THREE.PerspectiveCamera(48, 1, .05, 200); this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.maxPolarAngle = Math.PI * .49; this.controls.minDistance = 1; this.controls.maxDistance = 65;
    this.scene.add(new THREE.HemisphereLight('#f5f3e9', '#747c69', 2.3));
    const sun = new THREE.DirectionalLight('#fff2da', 3.4); sun.position.set(6, -8, 26);
    sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: .1, far: 70 });
    sun.shadow.normalBias = .04; this.scene.add(sun); this.scene.add(sun.target);
    sun.target.position.set(12, 6, 0);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: '#bdc3b7', roughness: 1 }));
    ground.position.z = -.23; ground.receiveShadow = true; this.scene.add(ground);
    this.model = new THREE.Group(); this.scene.add(this.model);
    this.drone = this.makeDrone(); this.scene.add(this.drone);
    this.mode = 'chase'; this.cutaway = true; this.trailVisible = true;
    this.wood = woodTexture(); this.wood.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.furnishings = new Furnishings(this.wood);
    this.cutawayHeight = 2.25;
    this.resize(); window.addEventListener('resize', () => this.resize());
  }

  resize() {
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.camera.aspect = window.innerWidth / window.innerHeight; this.updateLens();
  }

  updateLens() {
    this.camera.fov = this.mode === 'fpv' && this.data?.world.hfov
      ? THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(this.data.world.hfov / 2) / this.camera.aspect)) : 48;
    this.camera.updateProjectionMatrix();
  }

  makeDrone() {
    const drone = new THREE.Group();
    const shell = new THREE.MeshStandardMaterial({ color: '#e5e7df', metalness: .3, roughness: .34 });
    const carbon = new THREE.MeshStandardMaterial({ color: '#23302b', metalness: .5, roughness: .4 });
    const mesh = (geometry, material, x, y, z) => {
      const item = new THREE.Mesh(geometry, material); item.position.set(x, y, z); item.castShadow = true; drone.add(item); return item;
    };
    mesh(new THREE.BoxGeometry(.38, .28, .13), shell, 0, 0, .03);
    mesh(new THREE.BoxGeometry(.25, .2, .07), carbon, -.05, 0, .13);
    for (const angle of [Math.PI / 4, -Math.PI / 4]) {
      const arm = mesh(new THREE.BoxGeometry(.88, .055, .04), carbon, 0, 0, 0); arm.rotation.z = angle;
    }
    this.rotors = [];
    for (const x of [-.31, .31]) for (const y of [-.31, .31]) {
      const motor = mesh(new THREE.CylinderGeometry(.047, .05, .09, 16), carbon, x, y, .04); motor.rotation.x = Math.PI / 2;
      const rotor = mesh(new THREE.BoxGeometry(.36, .022, .008), carbon, x, y, .095); this.rotors.push(rotor);
      const disc = mesh(new THREE.CircleGeometry(.18, 24), new THREE.MeshBasicMaterial({ color: '#273a30', transparent: true, opacity: .07, side: THREE.DoubleSide }), x, y, .095);
      disc.castShadow = false;
    }
    for (const y of [-.17, .17]) {
      mesh(new THREE.BoxGeometry(.04, .025, .16), carbon, -.09, y, -.12);
      mesh(new THREE.BoxGeometry(.33, .025, .025), carbon, -.03, y, -.2);
    }
    mesh(new THREE.BoxGeometry(.09, .1, .085), carbon, .24, 0, -.025);
    const lens = mesh(new THREE.CylinderGeometry(.032, .032, .035, 16), new THREE.MeshStandardMaterial({ color: '#152e2c', metalness: .9, roughness: .13 }), .295, 0, -.025); lens.rotation.z = Math.PI / 2;
    mesh(new THREE.BoxGeometry(.012, .2, .018), new THREE.MeshBasicMaterial({ color: '#a1ffd0' }), .195, 0, .04);
    return drone;
  }

  load(data) {
    for (const child of [...this.model.children]) {
      child.traverse(item => { if (item.isSprite) item.material.map?.dispose(); item.geometry?.dispose(); item.material?.dispose(); });
      this.model.remove(child);
    }
    if (this.targetEffect) { this.scene.remove(this.targetEffect); this.targetEffect.dispose(); this.targetEffect = null; }
    this.walls = []; this.ceilings = []; this.labels = []; this.data = data;
    const targetName = data.world.mission.find(objective => objective.marker)?.marker + '_marker';
    for (const box of data.world.geometry) {
      const name = box.name;
      const mesh = this.furnishings.make(box); this.model.add(mesh);
      if (name === targetName) {
        mesh.material.emissive.copy(mesh.material.color); mesh.material.emissiveIntensity = .7; mesh.material.roughness = .25;
        this.targetEffect = new TargetEffects(box.center); this.scene.add(this.targetEffect);
      }
      if (/wall|lintel/.test(name)) this.walls.push(mesh);
      if (name === 'ceiling') this.ceilings.push(mesh);
    }
    for (const [name, [lo, hi]] of Object.entries(data.world.rooms)) {
      const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 96;
      const context = canvas.getContext('2d');
      context.fillStyle = '#1c302ae6'; context.beginPath(); context.roundRect(0, 0, 512, 96, 16); context.fill();
      context.fillStyle = '#e3ece4'; context.font = '500 42px sans-serif'; context.textAlign = 'center'; context.textBaseline = 'middle';
      context.fillText(name.replaceAll('_', ' ').toUpperCase(), 256, 50);
      const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
      const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
      label.position.set((lo[0] + hi[0]) / 2, lo[1] + .55, .3); label.scale.set(1.7, .32, 1);
      this.labels.push(label); this.model.add(label);
    }
    const positions = data.trace.map(t => t.position).flat();
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.trail = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: '#2e9e75', transparent: true, opacity: .75 }));
    this.model.add(this.trail);
    const bounds = Object.values(data.world.rooms);
    const maxX = Math.max(...bounds.map(b => b[1][0])), maxY = Math.max(...bounds.map(b => b[1][1]));
    this.center = new THREE.Vector3(maxX / 2, maxY / 2, 0);
    this.orbitPosition = new THREE.Vector3(maxX / 2 + maxX * .65, maxY / 2 - maxX * .7, maxX * .83);
    this.setMode(this.mode, true); this.setCutaway(this.cutaway);
  }

  setCutaway(value) {
    this.cutaway = value;
    for (const wall of this.walls || []) {
      const b = wall.userData;
      const lower = b.center[2] - b.half[2];
      const height = value && this.mode !== 'fpv' ? Math.max(0, Math.min(b.half[2] * 2, this.cutawayHeight - lower)) : b.half[2] * 2;
      wall.visible = height > 0; wall.scale.z = height / (b.half[2] * 2); wall.position.z = lower + height / 2;
    }
    for (const ceiling of this.ceilings || []) ceiling.visible = this.mode === 'fpv';
  }

  setMode(mode, snap = false) {
    this.mode = mode; this.controls.enabled = mode === 'orbit';
    this.updateLens();
    for (const label of this.labels || []) label.visible = mode === 'orbit';
    this.drone.visible = mode !== 'fpv'; this.setCutaway(this.cutaway);
    if (mode === 'orbit' && this.center) { this.camera.position.copy(this.orbitPosition); this.controls.target.copy(this.center); this.controls.update(); }
    this.snap = snap || mode !== 'orbit';
  }

  draw(sample, next, fraction, index, dt, playing, presentationTime) {
    const p = new THREE.Vector3().fromArray(sample.position).lerp(new THREE.Vector3().fromArray(next.position), fraction);
    const direction = new THREE.Vector3().fromArray(sample.camera).lerp(new THREE.Vector3().fromArray(next.camera), fraction).normalize();
    this.drone.position.copy(p);
    // A forward vector determines yaw/pitch, but cannot establish body roll.
    // Shortest-arc quaternions invent a large roll near a reversed heading.
    this.drone.rotation.set(0, -Math.atan2(direction.z, Math.hypot(direction.x, direction.y)),
      Math.atan2(direction.y, direction.x), 'ZYX');
    if (playing) this.rotors.forEach((rotor, i) => rotor.rotation.z += dt * 100 * (i % 2 ? -1 : 1));
    this.presentationTime = presentationTime ?? (this.presentationTime || 0) + dt;
    this.targetEffect?.update(this.presentationTime);
    if (this.trail) { this.trail.visible = this.trailVisible; this.trail.geometry.setDrawRange(0, index + 1); }
    if (this.mode === 'chase') {
      const behind = direction.clone().setZ(0).normalize().multiplyScalar(-4.8);
      const right = new THREE.Vector3(direction.y, -direction.x, 0).multiplyScalar(2.8);
      const desired = p.clone().add(behind).add(right).add(new THREE.Vector3(0, 0, 5.8));
      this.camera.position.lerp(desired, this.snap ? 1 : 1 - Math.exp(-dt * 4));
      const look = p.clone().add(direction.clone().multiplyScalar(.8));
      if (!this.lookTarget || this.snap) this.lookTarget = look;
      else this.lookTarget.lerp(look, 1 - Math.exp(-dt * 5));
      this.camera.lookAt(this.lookTarget); this.snap = false;
    } else if (this.mode === 'fpv') {
      this.camera.position.copy(p).add(direction.clone().multiplyScalar(.32)).add(new THREE.Vector3(0, 0, .1));
      this.camera.lookAt(this.camera.position.clone().add(direction));
    } else this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
