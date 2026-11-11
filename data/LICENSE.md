# xBD / xView2 pilot data

Original dataset: xBD / xView2 by Gupta et al., DIU / CMU SEI.
Imagery: Maxar Open Data Program.
Mirror: https://huggingface.co/datasets/kshitijrajsharma/xview2-xbd
Mirror revision: 079adbab22aa09633dfd50ece0dd0137431a9780
Original source: https://xview2.org/
Paper: Gupta et al. (2019), xBD: A Dataset for Assessing Building Damage from Satellite Imagery, https://arxiv.org/abs/1911.09296

Bundled image/mask subset: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
Licence: https://creativecommons.org/licenses/by-nc-sa/4.0/
Legal terms: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Retain attribution. Non-commercial use only. Adapted data must be shared under the same licence. No endorsement is implied. The MIT licence on application code does not relicense these assets.

Changes: selected at most eight scenes per event from one mirror training shard; resized images to 512 px, raster labels with nearest-neighbour; unknown labels set to ignore value 255. Exact provenance and checksum in catalog.json. No synthetic imagery is included.
