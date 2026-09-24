import * as THREE from 'three';

function lightTexture(star) {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
  const c = canvas.getContext('2d');
  const glow = c.createRadialGradient(64, 64, 0, 64, 64, 64);
  glow.addColorStop(0, '#fffcefff'); glow.addColorStop(.13, '#ffdab3dc'); glow.addColorStop(.4, '#ff6a703b'); glow.addColorStop(1, '#ff314000');
  c.fillStyle = glow; c.fillRect(0, 0, 128, 128);
  if (star) {
    c.fillStyle = '#fff8df'; c.beginPath();
    for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4, r = i % 2 ? 7 : 48; const x = 64 + Math.cos(a) * r, y = 64 + Math.sin(a) * r; if (!i) c.moveTo(x, y); else c.lineTo(x, y); }
    c.closePath(); c.fill();
  }
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace; return texture;
}

export class TargetEffects extends THREE.Group {
  constructor(position) {
    super(); this.position.fromArray(position); this.elapsed = 0;
    this.glowTexture = lightTexture(false); this.sparkTexture = lightTexture(true);
    const material = map => new THREE.SpriteMaterial({ map, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true, toneMapped: false });
    this.halo = new THREE.Sprite(material(this.glowTexture)); this.halo.scale.setScalar(1.35); this.halo.material.opacity = .55; this.add(this.halo);
    this.sparks = Array.from({ length: 18 }, (_, i) => {
      const sprite = new THREE.Sprite(material(this.sparkTexture)); sprite.userData.phase = i * 2.399963; this.add(sprite); return sprite;
    });
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(.33, .008, 6, 64), new THREE.MeshBasicMaterial({ color: '#ffdba0', transparent: true, opacity: .7, toneMapped: false }));
    this.ring.position.z = -.15; this.add(this.ring);
    this.add(new THREE.PointLight('#ff6779', .8, 2)); this.update(0);
  }

  update(seconds) {
    this.elapsed = seconds; this.halo.material.opacity = .44 + .12 * Math.sin(seconds * 2.8);
    for (let i = 0; i < this.sparks.length; i++) {
      const spark = this.sparks[i], phase = spark.userData.phase, cycle = (seconds * .32 + i / this.sparks.length) % 1;
      const angle = phase + seconds * .45, radius = .29 + .17 * Math.sin(phase * 3) ** 2;
      spark.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, -.17 + cycle * .9);
      spark.scale.setScalar(.07 + .1 * Math.sin(Math.PI * cycle) ** 3);
      spark.material.opacity = Math.sin(Math.PI * cycle) ** 2; spark.material.rotation = seconds * .4 + phase;
    }
    this.ring.rotation.z = seconds * .25;
  }

  dispose() {
    this.traverse(child => { child.geometry?.dispose(); child.material?.dispose(); });
    this.glowTexture.dispose(); this.sparkTexture.dispose();
  }
}
