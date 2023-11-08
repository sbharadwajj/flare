# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj

import nvdiffrast.torch as dr
import torch

class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    def __init__(self, device, near=1, far=1000, mesh=None):
        self.glctx = dr.RasterizeGLContext()
        self.device = device
        self.near = near
        self.far = far

    def set_near_far(self, dataset_train, samples, epsilon=0.1):
        mins = []
        maxs = []
        cameras = dataset_train.get_camera_mat()
        for cam in cameras:
            samples = samples.to(cam.device)
            samples_projected = cam.project(samples, depth_as_distance=True)
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())
        near, far = min(mins), max(maxs)
        near = near.to(self.device)
        far = far.to(self.device)
        self.near = near - (near * epsilon)
        self.far = far + (far * epsilon)
        
    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]
    
    @staticmethod
    def transform_pos_batch(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[..., 0:1])], axis=2)
        return torch.bmm(posw, t_mtx)

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device) 
    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000):

        projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                fy=camera.K[1,1],
                                                cx=camera.K[0,2],
                                                cy=camera.K[1,2],
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        gl_transform = torch.tensor([[1., 0,  0,  0],
                                    [0,  1., 0,  0],
                                    [0,  0, -1., 0],
                                    [0,  0,  0,  1.]], device=camera.device)

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt, Rt

    @staticmethod
    def to_gl_camera_batch(camera, resolution, n=1000, f=5000):

        # since we have a single camera (!) same intrinsics!! 
        intrinsics = camera[0].K 
        device = camera[0].device

        projection_matrix = Renderer.projection(fx=intrinsics[0,0],
                                                fy=intrinsics[1,1],
                                                cx=intrinsics[0,2],
                                                cy=intrinsics[1,2],
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=device)

        gl_transform = torch.tensor([[1., 0,  0,  0],
                                    [0,  1., 0,  0],
                                    [0,  0, -1., 0],
                                    [0,  0,  0,  1.]], device=device)
        
        p_l = []
        rt_gl = []
        for cam in camera:
            Rt = torch.eye(4, device=device)
            Rt[:3, :3] = cam.R
            Rt[:3, 3] = cam.t
            Rt_gl = gl_transform @ Rt
            P = projection_matrix @ Rt_gl
            # we transpose here instead of transposing while multiplying
            p_l.append(P.t().unsqueeze(0))
            rt_gl.append(Rt_gl.unsqueeze(0))
            
        return torch.cat(p_l, dim=0), torch.cat(rt_gl, dim=0)
    
    def render_batch(self, views, deformed_vertices, deformed_normals, channels, with_antialiasing, canonical_v, canonical_idx):
        """ Render G-buffers from a set of views.

        Args:
            views (List[Views]): 
        """
        batch_size = deformed_vertices.shape[0]
        # single fixed camera and for now we fix res also
        resolution = (512, 512)
        P_batch, Rt = Renderer.to_gl_camera_batch(views, resolution, n=self.near, f=self.far)

        canonical_verts_batch = canonical_v.unsqueeze(0).repeat(batch_size, 1, 1)
        deformed_vertices_clip_space = Renderer.transform_pos_batch(P_batch, deformed_vertices)
        idx = canonical_idx.int()
        face_idx = deformed_normals["face_idx"].int()
        rast, rast_out_db = dr.rasterize(self.glctx, deformed_vertices_clip_space, idx, resolution=resolution)

        view_dir = torch.cat([v.center.unsqueeze(0) for v in views], dim=0)
        view_dir = view_dir[:, None, None, :]
        gbuffer = {}

        # deformed points in G-buffer
        if "position" in channels or "depth" in channels:
            position, _ = dr.interpolate(deformed_vertices, rast, idx)
            gbuffer["position"] = dr.antialias(position, rast, deformed_vertices_clip_space, idx) if with_antialiasing else position

        # canonical points in G-buffer
        if "canonical_position" in channels:
            canonical_position, _ = dr.interpolate(canonical_verts_batch, rast, idx, rast_db=rast_out_db, diff_attrs='all')
            gbuffer["canonical_position"] = dr.antialias(canonical_position, rast, deformed_vertices_clip_space, idx) if with_antialiasing else canonical_position


        # normals in G-buffer
        if "normal" in channels:
            vertex_normals, _ = dr.interpolate(deformed_normals["vertex_normals"], rast, idx)
            face_normals, _ = dr.interpolate(deformed_normals["face_normals"], rast, face_idx)
            tangent_normals, _ = dr.interpolate(deformed_normals["tangent_normals"], rast, idx)
            gbuffer["vertex_normals"] = vertex_normals
            gbuffer["face_normals"] = face_normals
            gbuffer["tangent_normals"] = tangent_normals

        # mask of mesh in G-buffer
        if "mask" in channels:
            gbuffer["mask"] = (rast[..., -1:] > 0.).float() 

            
        # We store the deformed vertices in clip space, the transformed camera matrix and the barycentric coordinates
        # to antialias texture and mask after computing the color 
        gbuffer["rast"] = rast
        gbuffer["deformed_verts_clip_space"] = deformed_vertices_clip_space
        gbuffer["view_pos_gl"] = Rt[:, :3, 3]

        return gbuffer
    
    def render_canonical_mesh(self, view, meshes, color, with_antialiasing=True):
        """ Rasterizes the canonical mesh: for visualization purpose only

        Args:
            views (List[Views]): 
        """
        resolution = (512, 512)

        canon_verts = meshes.vertices
        idx = meshes.indices.int()
        P, Rt = Renderer.to_gl_camera(view, resolution, n=self.near, f=self.far)
        pos = Renderer.transform_pos(P, canon_verts)
        rast, _ = dr.rasterize(self.glctx, pos, idx, resolution=resolution)
        interpolated_color, _ = dr.interpolate(color[None, ...].type(torch.float32), rast, idx)

        del rast
        return interpolated_color[0]