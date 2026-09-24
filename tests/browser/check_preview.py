"""Browser-level replay controls and full-resolution canvas checks."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main(url):
    output = Path("report/px4-house")
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("#loading.hidden")
        expect(page.locator("#details")).not_to_be_visible()
        expect(page.locator(".mission-panel")).to_be_visible()
        page.get_by_role("button", name="Details", exact=True).click()
        expect(page.locator("#details")).to_be_visible()
        options = page.locator("#run option").evaluate_all(
            "(items)=>items.map(e=>({value:e.value,text:e.textContent}))"
        )
        index = page.request.get(url + "/data/index.json").json()
        successful = next(
            o["path"] for o in index if o.get("success") and not o.get("safety_failure")
        )
        page.locator("#run").select_option(successful)
        expect(page.locator("#outcome")).to_have_text("FINAL · COMPLETE")
        assert page.locator("canvas").evaluate("(e)=>[e.width,e.height]") == [1920, 1080]
        page.get_by_role("button", name="Pause replay", exact=True).click()
        recording = page.request.get(url + "/" + successful).json()
        objective_count = len(recording["world"]["mission"])
        replan = next(
            (
                d
                for d in recording["decisions"]
                if d["role"] == "control" and d["choice"] == "brake_and_replan"
            ),
            None,
        )
        if replan:
            page.locator("#scrub").evaluate(
                '(e,time)=>{e.value=time;e.dispatchEvent(new Event("input",{bubbles:true}))}',
                replan["time"] + 0.0001,
            )
            expect(page.locator("#now")).to_have_text("Replanning the route")
        before = page.locator("#position").inner_text()
        page.locator("#scrub").evaluate(
            '(e)=>{e.value=Number(e.max)*.45;e.dispatchEvent(new Event("input",{bubbles:true}))}'
        )
        page.wait_for_function(
            '(before)=>document.querySelector("#position").textContent!==before', arg=before
        )
        page.screenshot(path=str(output / "preview-chase.png"))
        page.get_by_role("button", name="Close details", exact=True).click()
        expect(page.locator("#details")).not_to_be_visible()
        expect(page.locator(".mission-panel")).to_be_visible()
        page.screenshot(path=str(output / "preview-quiet.png"))
        page.get_by_role("button", name="Details", exact=True).click()
        page.get_by_role("button", name="Explore house", exact=True).click()
        expect(page.get_by_role("button", name="Explore house", exact=True)).to_have_attribute(
            "aria-pressed", "true"
        )
        page.locator("#cutaway").uncheck()
        page.locator("#cutaway").check()
        page.locator("#trail").uncheck()
        page.locator("#trail").check()
        page.screenshot(path=str(output / "preview-orbit.png"))
        page.get_by_role("button", name="Drone view", exact=True).click()
        expect(page.locator("#view-note")).to_contain_text("not the recorded sensor feed")
        page.screenshot(path=str(output / "preview-drone-view.png"))
        page.get_by_role("button", name="Restart replay", exact=True).click()
        expect(page.get_by_role("button", name="Pause replay", exact=True)).to_be_visible()
        page.get_by_role("button", name="Pause replay", exact=True).click()
        page.locator("#scrub").evaluate(
            '(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}'
        )
        expect(page.locator("#stages")).to_have_text(
            f"{objective_count} / {objective_count} OBJECTIVES"
        )
        expect(page.locator("#now")).to_have_text("Mission complete")
        page.locator("#scrub").evaluate(
            '(e)=>{e.value=0;e.dispatchEvent(new Event("input",{bubbles:true}))}'
        )
        expect(page.locator("#stages")).to_have_text(f"0 / {objective_count} OBJECTIVES")
        expect(page.locator('[data-phase="navigate"]')).to_have_attribute("aria-current", "step")
        unsafe = next(o["path"] for o in index if o.get("success") and o.get("safety_failure"))
        page.locator("#run").select_option(unsafe)
        expect(page.locator("#outcome")).to_have_text("COMPLETE · SAFETY FAIL")
        expect(page.locator("#outcome")).to_have_class("failed")
        page.get_by_role("button", name="Pause replay", exact=True).click()
        page.locator("#scrub").evaluate(
            '(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}'
        )
        expect(page.locator("#now")).to_have_text("Objectives complete · safety check failed")
        failed = next(o["value"] for o in options if o["text"].endswith(" · timeout"))
        page.locator("#run").select_option(failed)
        expect(page.locator("#outcome")).to_have_text("TIMEOUT")
        expect(page.locator("#outcome")).to_have_class("failed")
        page.get_by_role("button", name="Pause replay", exact=True).click()
        page.locator("#scrub").evaluate(
            '(e)=>{e.value=Number(e.max)*.5;e.dispatchEvent(new Event("input",{bubbles:true}))}'
        )
        page.screenshot(path=str(output / "preview-latest-flight.png"))
        page.get_by_role("button", name="Enter fullscreen", exact=True).click()
        page.wait_for_function("document.fullscreenElement !== null")
        page.get_by_role("button", name="Enter fullscreen", exact=True).click()
        page.wait_for_function("document.fullscreenElement === null")
        page.locator("#run").select_option("data/scene.json")
        page.wait_for_selector("#loading.hidden")
        expect(page.locator("#outcome")).to_have_text("SCENE PREVIEW")
        page.screenshot(path=str(output / "preview-furnished.png"))
        page.get_by_role("button", name="Close details", exact=True).click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "preview-mobile.png"))
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        expect(page.get_by_role("button", name="Explore house", exact=True)).to_be_visible()
        page.get_by_role("button", name="Details", exact=True).click()
        expect(page.locator(".mission-panel")).not_to_be_visible()
        expect(page.get_by_role("button", name="Enter fullscreen", exact=True)).to_be_visible()
        assert page.locator(".views").bounding_box()["height"] < 65
        page.get_by_role("button", name="Close details", exact=True).click()
        expect(page.locator(".mission-panel")).to_be_visible()
        page.close()
        retina = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        retina.on("pageerror", lambda e: errors.append(str(e)))
        retina.goto(url, wait_until="networkidle")
        retina.wait_for_selector("#loading.hidden")
        retina.get_by_role("button", name="Details", exact=True).click()
        retina.locator("#run").select_option("data/scene.json")
        expect(retina.locator("#outcome")).to_have_text("SCENE PREVIEW")
        assert retina.locator("canvas").evaluate("(e)=>[e.width,e.height]") == [3840, 2160]
        retina.get_by_role("button", name="Close details", exact=True).click()
        retina.screenshot(path=str(output / "preview-4k.png"))
        orientation = retina.evaluate("""async () => {
          const {HouseView}=await import('./src/scene.js');
          const {Vector3}=await import('three');
          const container=document.createElement('div');
          const view=new HouseView(container);
          const result=[];
          for (const camera of [[-1,.01,.005],[-1,-.01,-.005],[.01,1,.005]]) {
            const sample={position:[1,1,1],camera};
            view.draw(sample,sample,0,0,.016,false);
            const forward=new Vector3(1,0,0).applyQuaternion(view.drone.quaternion);
            const up=new Vector3(0,0,1).applyQuaternion(view.drone.quaternion);
            result.push({error:forward.distanceTo(new Vector3(...camera).normalize()),up:up.z});
          }
          view.controls.dispose();view.renderer.dispose();
          return result;
        }""")
        assert all(r["error"] < 1e-6 and r["up"] > 0.99 for r in orientation), orientation
        assert not errors, errors
        browser.close()
        print(
            "PASS: successful/failed flight replay, time-correct objectives, scrubbing, play/restart, three cameras, cutaway/trail, scene switching, fullscreen, 1920×1080 and 3840×2160 canvases, mobile viewport, quiet default, details drawer, progress rewind, no browser errors."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    main(parser.parse_args().url.rstrip("/"))
