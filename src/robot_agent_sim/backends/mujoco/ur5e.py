from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]/"assets"/"robots"/"ur5e"
def ur5e_xml(scene="scene_000.xml"): return ROOT/"scenes"/scene
