import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

function fabricTexture() {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
  const c = canvas.getContext('2d'); c.fillStyle = '#d8d4c9'; c.fillRect(0, 0, 128, 128);
  for (let i = 0; i < 128; i += 3) {
    c.fillStyle = i % 2 ? '#c4c1b7' : '#eee9dc'; c.fillRect(i, 0, 1, 128);
    c.fillStyle = '#817e761d'; c.fillRect(0, i, 128, 1);
  }
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.repeat.set(8, 8);
  return texture;
}

// These finishes decorate the recorded collision boxes. They never alter the
// source scene, the flight path, or the observations originally sent to Jev.
export class Furnishings {
  constructor(wood) { this.wood = wood; this.fabric = fabricTexture(); }

  make(box) {
    const name = box.name;
    const finish = { color: new THREE.Color().setRGB(...box.color.slice(0, 3), THREE.SRGBColorSpace), roughness: .72 };
    if (name === 'floor') Object.assign(finish, { color: '#e4cda9', map: this.wood });
    else if (/wall|lintel|ceiling/.test(name)) finish.color = '#ede5d6';
    else if (/search_screen|partition/.test(name)) Object.assign(finish, { color: '#466655', map: this.fabric, roughness: .95 });
    else if (/fridge/.test(name)) Object.assign(finish, { color: '#c9d0c8', metalness: .58, roughness: .29 });
    else if (/handle|leg|rail|monitor|hob|stem/.test(name)) Object.assign(finish, { color: '#303d36', metalness: .35, roughness: .4 });
    else if (/target_stand/.test(name)) Object.assign(finish, { color: '#273d35', metalness: .25, roughness: .3 });
    else if (/toolbox/.test(name)) Object.assign(finish, { color: '#ac623e', metalness: .15, roughness: .4 });
    else if (/_book$/.test(name)) finish.color = ['#b88245', '#416a72', '#944d3b', '#cfbb93', '#445947'][box.body % 5];
    else if (/rug|cushion|sofa|chair_seat|chair_back|blanket/.test(name)) Object.assign(finish, { color: /rug/.test(name) ? '#ab9d82' : '#738b77', map: this.fabric, roughness: .96 });
    else if (/mattress|pillow/.test(name)) Object.assign(finish, { color: '#f1e6ce', map: this.fabric, roughness: .96 });
    else if (/table|shelf|wardrobe|counter|desk|console|sideboard|workbench|bed_/.test(name)) Object.assign(finish, { color: '#bd9668', map: this.wood, roughness: .5 });
    else if (/planter/.test(name)) finish.color = '#c3a786';
    else if (/plant_leaves/.test(name)) finish.color = '#4d7654';
    const size = box.half.map(v => v * 2);
    const rounded = /cushion|pillow|mattress|marker|fridge$|toolbox|target_stand/.test(name);
    const geometry = rounded ? new RoundedBoxGeometry(...size, 2, Math.min(.045, Math.min(...box.half) * .2)) : new THREE.BoxGeometry(...size);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial(finish));
    mesh.position.fromArray(box.center); mesh.castShadow = name !== 'floor' && name !== 'ceiling'; mesh.receiveShadow = true; mesh.userData = box;
    this.details(mesh, box);
    return mesh;
  }

  details(mesh, box) {
    const [x, y, z] = box.half, name = box.name;
    const piece = (size, position, color, wood = false) => {
      const detail = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial({ color, map: wood ? this.wood : null, roughness: .55 }));
      detail.position.set(...position); detail.castShadow = true; mesh.add(detail);
    };
    if (/search_screen|partition/.test(name)) {
      // Framed acoustic dividers: the solid occluding panel stays in place.
      for (const face of [-1, 1]) {
        for (const edge of [-1, 1]) piece([.022, .065, 2 * z], [face * (x + .012), edge * (y - .032), 0], '#aa8052', true);
        for (const edge of [-1, 1]) piece([.022, 2 * y, .075], [face * (x + .012), 0, edge * (z - .037)], '#aa8052', true);
        for (let yy = -y + .15; yy < y - .1; yy += .13) piece([.016, .025, 2 * z - .2], [face * (x + .01), yy, 0], '#b49061', true);
      }
      piece([2 * x + .055, 2 * y, .065], [0, 0, -z + .033], '#34453b');
    } else if (/fridge$/.test(name)) {
      piece([.012, 2 * y - .06, .018], [-x - .009, 0, .33], '#65796d');
      piece([.014, 2 * y - .05, .07], [-x - .009, 0, -z + .055], '#42554a');
      piece([.018, .13, .038], [-x - .015, .24, z - .16], '#34483e');
    } else if (/target_stand/.test(name)) {
      piece([2 * x + .01, 2 * y + .01, .035], [0, 0, z - .018], '#c7ab72');
      piece([2 * x + .01, 2 * y + .01, .035], [0, 0, -z + .018], '#a28655');
    } else if (/_book$/.test(name)) {
      for (const zz of [-z + .06, z - .06]) piece([2 * x - .016, .009, .009], [0, -y - .004, zz], '#ddcfb3');
    } else if (/toolbox/.test(name)) {
      piece([2 * x + .008, 2 * y + .008, .025], [0, 0, .065], '#63432d');
      piece([.16, .055, .025], [0, 0, z + .01], '#2b3a30');
      for (const xx of [-.14, .14]) piece([.025, .018, .055], [xx, -y - .009, .015], '#cfb777');
    } else if (/wardrobe$|console$|sideboard$|counter$/.test(name)) {
      if (x < y) piece([.01, .016, z * 1.85], [-x - .006, 0, 0], '#7d6445');
      else for (let xx = -x + .4; xx < x; xx += .5) piece([.012, .012, z * 1.85], [xx, y + .006, 0], '#7d6445');
    }
  }
}
