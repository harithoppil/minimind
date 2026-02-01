"""
LogicGrad: Geometry-Aware Optimizer for Bilinear Matrices

This optimizer is specifically designed for bilinear/logic tensor optimization,
such as the W_bilinear matrices in TempModel's attention mechanism.

Key features:
1. Tangent space projection: Keeps updates geometrically valid
2. Newton-Schulz whitening: Approximates natural gradient (curvature-aware)
3. Dimensional scaling: Stabilizes updates for high-dimensional matrices

Usage:
    from trainer.logicgrad import LogicGrad, create_dual_optimizer
    
    # Simple usage
    opt_logic, opt_adam = create_dual_optimizer(model)
    
    # In training loop
    opt_logic.zero_grad()
    opt_adam.zero_grad()
    loss.backward()
    opt_logic.step()
    opt_adam.step()
"""

import torch
import torch.optim as optim


@torch.compile
def project_and_whiten_3d(G, W, steps=5, eps=1e-7):
    """
    Project gradient onto tangent space of W and apply Newton-Schulz whitening.
    
    Args:
        G: Gradient tensor (n_heads, pope_dim, pope_dim)
        W: Weight tensor (same shape as G)
        steps: Number of Newton-Schulz iterations (0 = projection only)
        eps: Small constant for numerical stability
    
    Returns:
        Whitened gradient in tangent space
    """
    # 1. Project Gradient onto Tangent Space of W
    # G_orth = G - <G,W>/<W,W> * W
    g_flat = G.view(G.size(0), -1)
    w_flat = W.view(W.size(0), -1)
    
    dot = (g_flat * w_flat).sum(dim=1, keepdim=True)
    norm = (w_flat * w_flat).sum(dim=1, keepdim=True) + eps
    
    g_orth = g_flat - (dot / norm) * w_flat
    
    # Use float32 for compatibility (MPS doesn't support bfloat16 well)
    X = g_orth.view_as(G).float()

    if steps == 0:
        return X.to(G.dtype)

    # 2. Newton-Schulz Whitening (Orthogonalize the Update)
    # Normalize spectral norm estimate roughly
    X_norm = X.norm(dim=(1, 2), keepdim=True) + eps
    X = X / X_norm
    
    # Clamp to prevent numerical explosion
    X = torch.clamp(X, -10.0, 10.0)
    
    # Optimal coefficients for fast convergence
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        AT = X.transpose(-1, -2)
        A = X @ AT
        # Clamp intermediate values
        A = torch.clamp(A, -100.0, 100.0)
        B = b * A + c * A @ A
        B = torch.clamp(B, -100.0, 100.0)
        X = a * X + X @ B
        X = torch.clamp(X, -10.0, 10.0)
        
    return X.to(G.dtype)


class LogicGrad(optim.Optimizer):
    """
    Geometry-aware optimizer for bilinear matrices.
    
    Uses Riemannian optimization (tangent space projection) and 
    Newton-Schulz whitening (natural gradient approximation) for
    efficient optimization of bilinear transformation matrices.
    
    Args:
        params: Iterable of parameters (should be bilinear matrices only)
        lr: Learning rate (default: 0.02, higher than AdamW due to whitening)
        momentum: Momentum factor (default: 0.9)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    """
    
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 1. Project Gradient FIRST (Pre-Momentum)
                # This ensures the momentum buffer only accumulates valid tangent directions
                g = p.grad
                g_orth = project_and_whiten_3d(g, p.data, steps=0)  # steps=0 just does projection
                
                # 2. Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g_orth)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g_orth)
                
                if group['nesterov']:
                    g_update = g_orth.add(buf, alpha=momentum)
                else:
                    g_update = buf
                
                # 3. Whiten the UPDATE direction
                # This makes the step size adaptive based on curvature
                update_step = project_and_whiten_3d(g_update, p.data, steps=group['ns_steps'])
                
                # 4. Apply Update (Scaled by dimensionality for stability)
                scale = (p.size(-1) ** -0.5)
                p.data.add_(update_step, alpha=-lr * scale)


def create_dual_optimizer(model, adam_lr=3e-4, logic_lr=0.05, weight_decay=0.01):
    """
    Create dual optimizers: LogicGrad for W_bilinear, AdamW for everything else.
    
    Args:
        model: The model to optimize
        adam_lr: Learning rate for AdamW (standard params)
        logic_lr: Learning rate for LogicGrad (bilinear params)
        weight_decay: Weight decay for AdamW
    
    Returns:
        tuple: (opt_logic, opt_adam) - both optimizers
    """
    bilinear_params = []
    standard_params = []

    for name, param in model.named_parameters():
        if "W_bilinear" in name:
            print(f"🔹 LogicGrad managing: {name} {param.shape}")
            bilinear_params.append(param)
        else:
            standard_params.append(param)
    
    if len(bilinear_params) == 0:
        print("⚠️ No W_bilinear params found, using AdamW only")
        return None, torch.optim.AdamW(standard_params, lr=adam_lr, weight_decay=weight_decay)
    
    opt_logic = LogicGrad(bilinear_params, lr=logic_lr, momentum=0.9, ns_steps=5)
    opt_adam = torch.optim.AdamW(standard_params, lr=adam_lr, weight_decay=weight_decay)
    
    print(f"✅ Created dual optimizers: LogicGrad ({len(bilinear_params)} params) + AdamW ({len(standard_params)} params)")
    
    return opt_logic, opt_adam
