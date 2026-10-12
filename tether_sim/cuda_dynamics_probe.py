"""Bounded checks of existing compiled CUDA dynamics; caller holds GPU lease."""
def probe():
    import torch
    from .cuda_backend import CudaTether
    sim=CudaTether(5)
    sim.params[:,3]=0 # isolate thrust/gravity from linear drag
    sim.wind.zero_();sim.motors.zero_()
    actions=torch.zeros(5,4,device='cuda')
    actions[1]=1
    mix=torch.tensor([[1,-1,-1],[1,1,1],[-1,1,-1],[-1,-1,1]],device='cuda',dtype=torch.float32)
    for axis in range(3):actions[2+axis]=.5+.02*mix[:,axis]
    before=sim.position.clone()
    sim.step(actions,dt=.001,substeps=1,iterations=32)
    torch.cuda.synchronize()
    assert not sim.failed.any(), 'Compiled dynamics failed in diagnostic state'
    fall=sim.position[0]-before[0]
    expected=-9.81*.001*.001
    assert torch.allclose(fall[:,2],torch.full_like(fall[:,2],expected),atol=1e-7), 'Gravity/free fall sign or integration mismatch'
    assert torch.allclose(fall[:,:2],torch.zeros_like(fall[:,:2]),atol=1e-9), 'Unforced horizontal drift'
    assert sim.velocity[1,0,2]>sim.velocity[0,0,2], 'Positive motor command does not increase upward acceleration'
    rates=sim.orientation[2:,4:]
    assert (rates.diagonal()>0).all(), 'Positive mixer channel produces wrong angular-rate sign'
    assert torch.allclose(rates-torch.diag(rates.diagonal()),torch.zeros_like(rates),atol=1e-6), 'Mixer channel crosses axes'
    assert torch.allclose(torch.linalg.vector_norm(sim.orientation[:,:4],dim=1),torch.ones(5,device='cuda'),atol=1e-6), 'Quaternion normalization failed'
    return dict(status='pass',free_fall_z_m=fall[0,2].item(),expected_free_fall_z_m=expected,
        upward_velocity_delta_m_s=(sim.velocity[1,0,2]-sim.velocity[0,0,2]).item(),
        positive_channel_angular_rates_rad_s=rates.cpu().tolist(),
        scope='Existing compiled CUDA kernel, five isolated one-substep domains; not tether contact fidelity, long-horizon stability, calibration or hardware accuracy')
