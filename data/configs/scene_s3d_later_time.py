import sys
import numpy as np
import json

fields = [ "H2", "NH3", "NO", "N2O", "O2", "H", "O", "OH", "HO2", "H2O", "H2O2", "NO2", "HNO", "N", "NNH", "NH2", "NH", "H2NO", "N2", "temp", "pressure", "velx", "vely", "velz" ]

with open("scene_s3d_later_time.json", "r") as f:
  basejson = json.load(f)

del basejson["view"]["volume"]["scalarMappingRange"]

for f in fields:
  basejson["dataSource"][0]["name"] = f
  basejson["dataSource"][0]["offset"] = fields.index(f) * 8 * 3456 * 960 * 2560

  with open(f"scene_s3d_later_time/field_{f}.json", "w") as f:
    json.dump(basejson, f, indent=2)
