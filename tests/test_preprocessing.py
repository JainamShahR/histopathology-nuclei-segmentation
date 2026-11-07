import numpy as np
from PIL import Image
from histoseg.monuseg import xml_to_binary_mask
from histoseg.tiles import iter_tiles, tile_starts
from histoseg.data import PairedImageDataset


def test_xml_polygons_rasterized(tmp_path):
    xml = tmp_path / 'example.xml'
    xml.write_text('''<Annotations><Annotation><Regions><Region><Vertices>
        <Vertex X="3" Y="3"/><Vertex X="11" Y="3"/>
        <Vertex X="11" Y="11"/><Vertex X="3" Y="11"/>
        </Vertices></Region></Regions></Annotation></Annotations>''')
    mask = xml_to_binary_mask(xml, 16, 16)
    assert mask.shape == (16, 16)
    assert mask[6, 6] == 255
    assert mask[0, 0] == 0


def test_tiling_covers_awkward_dimensions_and_edges():
    arr = np.zeros((213, 257, 3), dtype=np.uint8)
    coverage = np.zeros(arr.shape[:2], dtype=np.uint8)
    for y, x, tile in iter_tiles(arr, tile_size=128, stride=96):
        assert tile.shape == (128, 128, 3)
        coverage[y:min(y+128, 213), x:min(x+128, 257)] += 1
    assert np.all(coverage > 0)
    assert tile_starts(75, 128, 64) == [0]


def test_paired_dataset_loads_binary_masks(tmp_path):
    d = tmp_path / 'train'
    (d / 'images').mkdir(parents=True)
    (d / 'masks').mkdir()
    Image.fromarray(np.zeros((20, 21, 3), dtype=np.uint8)).save(d / 'images' / 'x.png')
    m = np.zeros((20, 21), dtype=np.uint8)
    m[4:8, 4:8] = 255
    Image.fromarray(m).save(d / 'masks' / 'x.png')
    ds = PairedImageDataset(tmp_path, 'train', 32)
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert y.shape == (1, 32, 32)
    assert set(y.unique().tolist()) == {0.0, 1.0}
