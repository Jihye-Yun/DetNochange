import os
import glob
import imageio
import numpy as np


class CXR(object):
    CLASS = ["change", "nochange"]


class CXRPair(object):

    def __init__(self, path):
        print (f"\n>>> dataset: {path}")

        self.data = []
        for _idx_class, _class in enumerate(CXR.CLASS):
            imgs = glob.glob(os.path.join(path, _class, "*_diff.png"))

            print (f"    - {_idx_class}: {_class} >>> {len(imgs)} pairs")

            for _img in imgs:
                base = _img.replace("_diff.png", "_base.png")
                if not os.path.exists(base): continue

                registered_fu = _img.replace("_diff.png", "_registered.png")
                if not os.path.exists(registered_fu): continue

                self.data.append({"diff": _img, "base": base, "registered_fu": registered_fu, "class": _idx_class})

        print (f"    => Loaded dataset: {len(self.data)} pairs")


    def load_data(self, idx):
        def preprocess_image(in_img, mode):
            output_img = in_img/255.
            if mode == 'diff': return abs(output_img*2.-1.)
            return output_img

        def load_image(path, mode):
            img = np.squeeze(imageio.imread(path))
            return np.expand_dims(preprocess_image(img, mode), axis=2)

        output_class = np.zeros(shape=(2,), dtype=int)
        output_class[self.data[idx]["class"]] = 1

        return np.concatenate([load_image(self.data[idx]["base"], 'org'),
                               load_image(self.data[idx]["registered_fu"], 'org'),
                               load_image(self.data[idx]["diff"], 'diff')], axis=2), output_class

    def __len__(self):
        return len(self.data)
