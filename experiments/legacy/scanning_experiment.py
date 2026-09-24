"""Follow-up: Jev chooses one translation or stop-and-look action per decision."""
from jev_drone.evidence import snapshot_sources
import argparse
import json
from pathlib import Path

from experiments.legacy.inspection_experiment import InspectionExperiment, InspectionNavigation
from jev_drone.gateway import JevGateway, load_credential
from experiments.legacy.jev_navigation import LOOK


class ScanningNavigation(InspectionNavigation):
    def questions(self):
        questions=super().questions()
        questions.pop("look",None)
        movement=questions["movement"]
        movement["instructions"] += "\nYou choose ONE combined camera/flight action. A look_DIRECTION action stops translation and points the camera there; fresh observations arrive on the next decision. If a needed movement direction is UNKNOWN, choose its look action first. NEVER translate into UNKNOWN or BLOCKED directions. If at a search viewpoint, use look actions to cover missing_views_here. If near a remembered target, look toward its surface to confirm it."
        movement["criteria"]={
            **{name:f"Move {name} at 1 m/s. Only choose if the named {name} clearance in the current observation is OPEN, never UNKNOWN/BLOCKED. Camera stays aimed as before." for name in LOOK},
            **{"look_"+name:f"Brake to stop and aim the camera {name}. Observe that unknown direction or inspect the target/search viewpoint; do not translate." for name in LOOK},
            "brake":"Stop and keep camera aim. Use when arrived and no further inspection is needed, or while stopping."}
        return questions

    def decode(self,result):
        if result.get("error"):
            result["action"]="brake"
            return result
        selected=result["answers"]["movement"]["choice"]
        result["selected_action"]=selected
        result["action"]="brake" if selected.startswith("look_") else selected
        result["look"]=selected.removeprefix("look_") if selected.startswith("look_") else self.look
        result["intent"]=result["answers"]["intent"]["choice"]
        return result


class ScanningExperiment(InspectionExperiment):
    def __init__(self,args,gateway):
        super().__init__(args,gateway)
        self.nav=ScanningNavigation(gateway,args.memory,args.limited,args.degraded,args.seed)

    def run(self):
        result=super().run()
        result["joint_scan_control"]=True
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=["near","far","occluded","absent"],required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--provider",choices=["openrouter","vercel"],default="openrouter")
    parser.add_argument("--memory",choices=["history","map","recovery"],default="history")
    parser.add_argument("--key-file",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.limited,args.uncertain,args.degraded=True,True,False
    args.architecture,args.planner="task","jev"
    args.seconds,args.budget=210.,.25
    key=load_credential(args.provider,args.key_file)
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    gateway=JevGateway(args.provider,key)
    try:
        result=ScanningExperiment(args,gateway).run()
    finally:
        gateway.close()
    (args.out/"episode.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("case","seed","status","simulation_seconds","role_counts","cost_usd","error")}),flush=True)


if __name__=="__main__": main()
