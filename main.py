import os

os.system(
    "pip install torch==2.5.1 && pip install packaging hydra-core scikit-learn wheel iopath onnxruntime rembg && pip install transformers==4.45.2 && pip install git+https://github.com/bsushsiwba/grounded-sam-pypi.git"
)
print("Dependencies installed successfully.")

from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
import os
from samutils import *

base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "shirt": "shirt",
            "trouser": "trouser",
        }
    )
)

print("GroundedSAM2 model initialized successfully.")

# read flag.txt or initialize with 1
with open("flag.txt", "w") as f:
    f.write("1")

while bool(int(open("flag.txt", "r").read())):
    try:
        with open("process.txt", "r") as f:
            f.read()
        # segmenting belly first
        samutils_segment(base_model)
        os.remove("process.txt")
    except:
        pass
