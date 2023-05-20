import trimesh
import trimesh.viewer
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.cluster
import scipy.optimize
import sklearn.cluster
import ctypes
import copy
import os
import sys
from sksparse.cholmod import cholesky
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, vstack, hstack
totals = []
for j in range(1,3):
    path = f"face{j}"

    def as_mesh(scene_or_mesh):
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()])
        else:
            mesh = scene_or_mesh
        return mesh

    verts = []
    for i in range(55):
        mesh = as_mesh(trimesh.load_mesh(os.path.join(f"{path}", f"{path}-{i:02}.obj")))
        verts.append(mesh)
    final = []
    finalmeshes = []
    for i in range(len(verts)):
        if not any([np.all(np.array(verts[i].vertices)==np.array(mesh.vertices)) for mesh in finalmeshes]):
            final.append(i)
            finalmeshes.append(verts[i])
    totals.append(set(final))
final = totals[0].intersection(totals[1])
for j in range(1,3):
    path = f"face{j}"
    def as_mesh(scene_or_mesh):
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()])
        else:
            mesh = scene_or_mesh
        return mesh

    verts = []
    for i in range(55):
        mesh = as_mesh(trimesh.load_mesh(os.path.join(f"{path}", f"{path}-{i:02}.obj")))
        verts.append(mesh)
    for i, ind in enumerate(final):
        verts[ind].export(f"{path}-new/{path}-{i:02}.obj")