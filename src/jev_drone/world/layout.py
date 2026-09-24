"""Known architectural geometry; contains no furniture or target locations."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    rooms: dict
    doors: dict
    connections: dict

    def room_at(self, position):
        for name, (lo, hi) in self.rooms.items():
            if all(lo[i] <= position[i] <= hi[i] for i in range(3)):
                return name
        return "unknown"

    @property
    def bounds(self):
        return (np.min([v[0] for v in self.rooms.values()], axis=0),
                np.max([v[1] for v in self.rooms.values()], axis=0))

    def door_axis(self, name):
        first, second = (self.rooms[r] for r in self.connections[name])
        for axis in (0, 1):
            if first[1][axis] == second[0][axis] or second[1][axis] == first[0][axis]:
                return axis
        raise ValueError("Door must connect rooms sharing a wall: "+name)

    def door_direction(self, name, destination):
        axis = self.door_axis(name)
        lo, hi = self.rooms[destination]
        return 1 if (lo[axis]+hi[axis])/2 > self.doors[name][axis] else -1

    def neighbors(self):
        graph = {room: [] for room in self.rooms}
        for name, (first, second) in self.connections.items():
            graph[first].append({"room": second, "door": name})
            graph[second].append({"room": first, "door": name})
        return graph

    def search_viewpoints(self):
        """Architectural candidates only; furniture/visibility must still be observed."""
        views = {}
        for room, (lo, hi) in self.rooms.items():
            positions = {}
            for ix, x in enumerate((lo[0]+1.5, hi[0]-1.5)):
                for iy, y in enumerate((lo[1]+1., hi[1]-1.)):
                    positions[f'{ix}_{iy}'] = (x,y)
            # Corners alone can all be occupied or occluded by unknown furniture.
            positions.update(edge_west=(lo[0]+1.5,(lo[1]+hi[1])/2),
                edge_east=(hi[0]-1.5,(lo[1]+hi[1])/2),
                edge_south=((lo[0]+hi[0])/2,lo[1]+1.),
                edge_north=((lo[0]+hi[0])/2,hi[1]-1.))
            for name, (x,y) in positions.items():
                # Leave room for the observed swept volume and stopping margin.
                for z in (1.3, round(min(2.9, hi[2]-1.), 2)):
                    views[f'view_{room}_{name}_{z}'] = {'room':room,'position':[x,y,z]}
        return views


SMALL = Layout(
    {"living": ((0, 0, 0), (6, 8, 4)), "bedroom": ((6, 0, 0), (12, 4, 4)),
     "study": ((6, 4, 0), (12, 8, 4))},
    {"living_bedroom": (6, 2, 1.3), "living_study": (6, 6, 1.3),
     "bedroom_study": (9, 4, 1.3)},
    {"living_bedroom": ("living", "bedroom"), "living_study": ("living", "study"),
     "bedroom_study": ("bedroom", "study")})

HOUSE = Layout(
    {name: ((6*x, 6*y, 0), (6*x+6, 6*y+6, 3.2))
     for y, row in enumerate((("entry", "living", "dining", "kitchen"),
                              ("study", "bedroom", "library", "workshop")))
     for x, name in enumerate(row)},
    {"entry_living": (6, 3, 1.3), "living_dining": (12, 3, 1.3),
     "dining_kitchen": (18, 3, 1.3), "study_bedroom": (6, 9, 1.3),
     "bedroom_library": (12, 9, 1.3), "library_workshop": (18, 9, 1.3),
     "entry_study": (3, 6, 1.3), "living_bedroom": (9, 6, 1.3),
     "dining_library": (15, 6, 1.3), "kitchen_workshop": (21, 6, 1.3)},
    {name: tuple(name.split("_")) for name in (
        "entry_living", "living_dining", "dining_kitchen", "study_bedroom",
        "bedroom_library", "library_workshop", "entry_study", "living_bedroom",
        "dining_library", "kitchen_workshop")})

HOUSE_CASES = ("house_far", "house_hidden", "house_sequence", "house_reverse", "house_blocked",
               "house_search", "house_search_absent")


def for_case(case):
    return HOUSE if case in HOUSE_CASES else SMALL
