import trimesh
import nibabel as nib
from scipy import ndimage
import numpy as np

inputFIlePath = '/home/x17wang/Data/prm001/prm001_40w_Rwhite_2.stl'

m = trimesh.load(inputFIlePath) 

v = m.voxelized(pitch=0.5)  #Return a Voxel object representing the current mesh discretized into voxels at the specified pitch, pitch: float, the edge length of a single voxel

v = v.fill(method='holes')

cog = np.sum(v.points, axis=0)
cog /= np.size(v.points, axis=0) # The center coordinate(x,y,z)

# Convert mesh to binary image
outimage = np.zeros((2*int(np.round(cog[0]))+1, 2*int(np.round(cog[1]))+1, 2*int(np.round(cog[2]))+1), dtype=np.int16)

for i in range(np.size(v.points, axis=0)):
  outimage[int(np.round(v.points[i,0])), int(np.round(v.points[i,1])), int(np.round(v.points[i,2]))] = 1

# Save binary image in a nifti file
nii = nib.load('/home/x17wang/Data/prm001/prm001_40w_Rwhite.nii')
save_path = '/home/x17wang/Exp/prm001/test_prm001_40w_Rwhite_2.nii.gz'

#img = nib.Nifti1Image(outimage, nii.affine)
aff = np.eye(3)
aff[0,0] = 0.5
aff[1,1] = 0.5
aff[2,2] = 0.5

img = nib.Nifti1Image(outimage, aff)
nib.save(img, save_path)
