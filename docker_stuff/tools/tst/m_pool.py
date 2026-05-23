
def foo_cluster(tp):
    from pyballistics import ozvb_termo
    omq, delta = tp
    
    
    d = 0.05
    q = 4
    omega = omq * q
    opts = {
        'powders': [
            {'omega': omega, 'dbname': '12/1 Тр'}],
        'init_conditions': {
            'q': q,
            'd':  d,
            'W_0': omega / delta,
            'p_0': 30e6,
            'phi_1': 1.02},
        'igniter': {'p_ign_0': 1e6},
        'meta_termo': {'dt': 2.5e-06 , 'method': 'rk4' },
        'stop_conditions': {
            'x_p': 35*d, 
            'steps_max': 300_000, 
            'p_max': 380e6, 
            'v_p': 400 }
    }
        
    res = ozvb_termo(opts)
    if res['stop_reason'] == 'p_max': return (1,0,0)
    if res['stop_reason'] == 'v_p':   return (0,0.8,0)
    if res['stop_reason'] == 'x_p':   return (0,0,1)
    return (0,0,0)
