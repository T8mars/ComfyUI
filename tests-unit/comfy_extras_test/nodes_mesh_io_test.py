import base64
import json
import struct
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_api.latest import Types
from comfy_extras.mesh3d.fileio import gltf_read, mesh_file_read
from comfy_extras.nodes_mesh_io import File3DToMesh, _merge_primitives

with patch.dict("sys.modules", {"server": MagicMock()}):
    from comfy_extras.nodes_save_3d import save_glb


def _warn(_key, _msg):
    pass


def _run_node(data: bytes, fmt: str) -> Types.MESH:
    out = File3DToMesh.execute(Types.File3D(BytesIO(data), file_format=fmt))
    return out.args[0]


def _data_uri(raw: bytes) -> str:
    return "data:application/octet-stream;base64," + base64.b64encode(raw).decode()


class TestGLBRoundTrip:
    def test_full_attribute_round_trip(self):
        vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        uvs = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32)
        normals = torch.nn.functional.normalize(torch.tensor(
            [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32), dim=1)
        tangents = torch.tensor([[1, 0, 0, 1]] * 4, dtype=torch.float32)
        texture_px = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

        glb = save_glb(vertices, faces, None, None, uvs=uvs, vertex_colors=colors,
                       texture_image=Image.fromarray(texture_px), normals=normals, tangents=tangents)
        mesh = _run_node(glb, "glb")

        assert torch.allclose(mesh.vertices[0], vertices, atol=1e-6)
        assert torch.equal(mesh.faces[0], faces)
        assert torch.allclose(mesh.uvs[0], uvs, atol=1e-6)
        assert torch.allclose(mesh.vertex_colors[0], colors, atol=1e-6)
        assert torch.allclose(mesh.normals[0], normals, atol=1e-5)
        assert torch.allclose(mesh.tangents[0], tangents, atol=1e-5)
        expected_texture = torch.from_numpy(texture_px.astype(np.float32) / 255.0)
        assert torch.allclose(mesh.texture[0], expected_texture, atol=1e-6)

    def test_mesh_to_file3d_output_parses(self):
        with patch.dict("sys.modules", {"server": MagicMock()}):
            from comfy_extras.nodes_save_3d import MeshToFile3D
        vertices = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
        faces = torch.tensor([[[0, 1, 2]]], dtype=torch.int64)
        file_3d = MeshToFile3D.execute(Types.MESH(vertices, faces)).args[0]
        mesh = File3DToMesh.execute(file_3d).args[0]
        assert torch.allclose(mesh.vertices, vertices, atol=1e-6)
        assert torch.equal(mesh.faces, faces)


def _gltf_bytes(gltf: dict) -> bytes:
    return json.dumps(gltf).encode()


def _positions_buffer(points) -> tuple[str, int]:
    raw = np.asarray(points, np.float32).tobytes()
    return _data_uri(raw), len(raw)


class TestGLTFSemantics:
    def _base(self, points, mode, extra_nodes=None, node_extra=None):
        uri, length = _positions_buffer(points)
        node = {"mesh": 0}
        node.update(node_extra or {})
        return {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": uri, "byteLength": length}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": length}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": mode}]}],
            "nodes": [node] + (extra_nodes or []),
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }

    def test_triangle_strip_and_fan(self):
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        strip = _run_node(_gltf_bytes(self._base(points, 5)), "gltf")
        assert strip.faces[0].tolist() == [[0, 1, 2], [3, 2, 1]]
        fan = _run_node(_gltf_bytes(self._base(points, 6)), "gltf")
        assert fan.faces[0].tolist() == [[0, 1, 2], [0, 2, 3]]

    def test_node_transform_baked(self):
        gltf = self._base([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 4,
                          node_extra={"translation": [10, 0, 0], "scale": [2, 2, 2]})
        mesh = _run_node(_gltf_bytes(gltf), "gltf")
        assert mesh.vertices[0].tolist() == [[10, 0, 0], [12, 0, 0], [10, 2, 0]]

    def test_negative_scale_flips_winding(self):
        gltf = self._base([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 4, node_extra={"scale": [-1, 1, 1]})
        mesh = _run_node(_gltf_bytes(gltf), "gltf")
        assert mesh.faces[0].tolist() == [[2, 1, 0]]

    def test_normalized_ubyte_colors(self):
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        pos_raw = np.asarray(points, np.float32).tobytes()
        col_raw = np.asarray([[0, 128, 255]] * 3, np.uint8).tobytes()
        raw = pos_raw + col_raw
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": _data_uri(raw), "byteLength": len(raw)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_raw)},
                {"buffer": 0, "byteOffset": len(pos_raw), "byteLength": len(col_raw)},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC3", "normalized": True},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
        }
        mesh = _run_node(_gltf_bytes(gltf), "gltf")
        assert torch.allclose(mesh.vertex_colors[0],
                              torch.tensor([[0.0, 128 / 255, 1.0]] * 3), atol=1e-6)

    def test_sparse_accessor(self):
        idx_raw = np.asarray([1], np.uint16).tobytes()
        val_raw = np.asarray([[5, 5, 5]], np.float32).tobytes()
        raw = idx_raw + b"\x00\x00" + val_raw
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": _data_uri(raw), "byteLength": len(raw)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_raw)},
                {"buffer": 0, "byteOffset": 4, "byteLength": len(val_raw)},
            ],
            "accessors": [{
                "componentType": 5126, "count": 3, "type": "VEC3",
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": 0, "componentType": 5123},
                    "values": {"bufferView": 1},
                },
            }],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
        }
        _, buffers, prims = gltf_read.load_gltf(_gltf_bytes(gltf), None, _warn)
        assert prims[0]["positions"].tolist() == [[0, 0, 0], [5, 5, 5], [0, 0, 0]]

    def test_interleaved_byte_stride(self):
        # position + uv interleaved in one 20-byte-stride view
        interleaved = np.zeros((3,), dtype=[("pos", np.float32, 3), ("uv", np.float32, 2)])
        interleaved["pos"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        interleaved["uv"] = [[0, 0], [1, 0], [0, 1]]
        raw = interleaved.tobytes()
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": _data_uri(raw), "byteLength": len(raw)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(raw), "byteStride": 20}],
            "accessors": [
                {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC2"},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
        }
        mesh = _run_node(_gltf_bytes(gltf), "gltf")
        assert mesh.uvs[0].tolist() == [[0, 0], [1, 0], [0, 1]]

    def test_draco_refused(self):
        gltf = self._base([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 4)
        gltf["extensionsRequired"] = ["KHR_draco_mesh_compression"]
        with pytest.raises(ValueError, match="compressed-geometry"):
            _run_node(_gltf_bytes(gltf), "gltf")

    def test_gltf_1_rejected(self):
        gltf = self._base([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 4)
        gltf["asset"] = {"version": "1.0"}
        with pytest.raises(ValueError, match="glTF asset version"):
            _run_node(_gltf_bytes(gltf), "gltf")

    def test_gpu_instancing_expanded(self):
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        translations = np.asarray([[0, 0, 0], [10, 0, 0]], np.float32)
        pos_raw = np.asarray(points, np.float32).tobytes()
        raw = pos_raw + translations.tobytes()
        gltf = {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["EXT_mesh_gpu_instancing"],
            "buffers": [{"uri": _data_uri(raw), "byteLength": len(raw)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_raw)},
                {"buffer": 0, "byteOffset": len(pos_raw), "byteLength": len(raw) - len(pos_raw)},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"mesh": 0, "extensions": {"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 1}}}}],
            "scenes": [{"nodes": [0]}],
        }
        mesh = _run_node(_gltf_bytes(gltf), "gltf")
        assert mesh.vertices.shape == (1, 6, 3)
        assert mesh.vertices[0, 3].tolist() == [10, 0, 0]
        assert mesh.faces[0].tolist() == [[0, 1, 2], [3, 4, 5]]

    def test_webp_texture_via_extension(self):
        px = np.full((2, 2, 3), 128, np.uint8)
        buf = BytesIO()
        Image.fromarray(px).save(buf, "WEBP", lossless=True)
        webp_uri = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        pos_raw = np.asarray(points, np.float32).tobytes()
        uv_raw = np.asarray([[0, 0], [1, 0], [0, 1]], np.float32).tobytes()
        raw = pos_raw + uv_raw
        gltf = {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["EXT_texture_webp"],
            "buffers": [{"uri": _data_uri(raw), "byteLength": len(raw)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_raw)},
                {"buffer": 0, "byteOffset": len(pos_raw), "byteLength": len(uv_raw)},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            ],
            "images": [{"uri": webp_uri}],
            "textures": [{"extensions": {"EXT_texture_webp": {"source": 0}}}],
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
        }
        mesh = _run_node(_gltf_bytes(gltf), "gltf")
        assert mesh.texture is not None
        assert torch.allclose(mesh.texture[0], torch.full((2, 2, 3), 128 / 255), atol=1e-6)


class TestOBJ:
    def test_quad_uv_normals(self):
        obj = (
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "vn 0 0 1\n"
            "f 1/1/1 2/2/1 3/3/1 4/4/1\n"
        )
        prim = mesh_file_read.load_obj(obj.encode())
        assert prim["faces"].tolist() == [[0, 1, 2], [0, 2, 3]]
        assert prim["uvs"].tolist() == [[0, 1], [1, 1], [1, 0], [0, 0]]
        assert prim["normals"].tolist() == [[0, 0, 1]] * 4

    def test_negative_indices_and_dedup(self):
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nf 1 2 3\n"
        prim = mesh_file_read.load_obj(obj.encode())
        assert prim["positions"].shape == (3, 3)
        assert prim["faces"].tolist() == [[0, 1, 2], [0, 1, 2]]

    def test_vertex_colors_extension(self):
        obj = "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n"
        prim = mesh_file_read.load_obj(obj.encode())
        assert prim["colors"].tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_vertex_colors_srgb_to_linear(self):
        obj = "v 0 0 0 0.5 0.5 0.5\nv 1 0 0 0.5 0.5 0.5\nv 0 1 0 0.5 0.5 0.5\nf 1 2 3\n"
        prim = mesh_file_read.load_obj(obj.encode())
        assert prim["colors"] == pytest.approx(np.full((3, 3), 0.21404114), abs=1e-6)

    def test_backslash_line_continuation(self):
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 \\\n2 3\n"
        prim = mesh_file_read.load_obj(obj.encode())
        assert prim["faces"].tolist() == [[0, 1, 2]]

    def test_mixed_missing_normals_dropped(self):
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3\n"
        prim = mesh_file_read.load_obj(obj.encode())
        assert prim["normals"] is None

    def test_no_faces_raises(self):
        with pytest.raises(ValueError, match="no faces"):
            mesh_file_read.load_obj(b"v 0 0 0\nv 1 0 0\n")


class TestSTL:
    def test_binary(self):
        tri = struct.pack("<3f", 0, 0, 1) + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0) + b"\x00\x00"
        data = b"\x00" * 80 + struct.pack("<I", 1) + tri
        prim = mesh_file_read.load_stl(data)
        assert prim["positions"].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        assert prim["faces"].tolist() == [[0, 1, 2]]
        assert prim["normals"].tolist() == [[0, 0, 1]] * 3

    def test_ascii(self):
        data = (
            "solid cube\n"
            " facet normal 0 0 1\n"
            "  outer loop\n"
            "   vertex 0 0 0\n"
            "   vertex 1 0 0\n"
            "   vertex 0 1 0\n"
            "  endloop\n"
            " endfacet\n"
            "endsolid cube\n"
        ).encode()
        prim = mesh_file_read.load_stl(data)
        assert prim["positions"].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        assert prim["normals"].tolist() == [[0, 0, 1]] * 3

    def test_binary_with_trailing_junk(self):
        tri = struct.pack("<3f", 0, 0, 1) + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0) + b"\x00\x00"
        data = b"\x00" * 80 + struct.pack("<I", 1) + tri + b"JUNKJUNK"
        prim = mesh_file_read.load_stl(data)
        assert prim["faces"].tolist() == [[0, 1, 2]]

    def test_binary_color_extension(self):
        header = b"COLOR=" + bytes([255, 255, 255, 255]) + b"\x00" * 70
        attr_per_face = struct.pack("<H", 31)
        tri = struct.pack("<3f", 0, 0, 1) + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0) + attr_per_face
        data = header + struct.pack("<I", 1) + tri
        prim = mesh_file_read.load_stl(data)
        assert prim["colors"] == pytest.approx(np.tile([1.0, 0.0, 0.0], (3, 1)), abs=1e-6)

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="STL"):
            mesh_file_read.load_stl(b"not a mesh at all")


class TestMergeAndNode:
    def test_merge_pads_missing_attributes(self):
        a = {"positions": np.zeros((3, 3), np.float32), "faces": np.array([[0, 1, 2]], np.int64),
             "uvs": np.ones((3, 2), np.float32), "colors": None, "normals": np.ones((3, 3), np.float32),
             "tangents": None, "material": None}
        b = {"positions": np.ones((3, 3), np.float32), "faces": np.array([[0, 1, 2]], np.int64),
             "uvs": None, "colors": np.zeros((3, 4), np.float32), "normals": None,
             "tangents": None, "material": None}
        merged = _merge_primitives([a, b])
        assert merged["vertices"].shape == (6, 3)
        assert merged["faces"].tolist() == [[0, 1, 2], [3, 4, 5]]
        assert merged["uvs"][3:].tolist() == [[0, 0]] * 3
        assert merged["colors"][:3].tolist() == [[1, 1, 1, 1]] * 3
        assert merged["normals"] is None

    def test_fbx_rejected(self):
        with pytest.raises(ValueError, match="FBX|fbx"):
            File3DToMesh.execute(Types.File3D(BytesIO(b"whatever"), file_format="fbx"))

    def test_sniff_glb_without_format(self):
        vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        glb = save_glb(vertices, faces)
        mesh = File3DToMesh.execute(Types.File3D(BytesIO(glb))).args[0]
        assert mesh.vertices.shape == (1, 3, 3)
