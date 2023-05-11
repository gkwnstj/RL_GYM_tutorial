import torch

def saveasONNX(actor_model, critic_model, device):
    actor_model.eval()
    critic_model.eval()

    dummy_state = torch.randn(1, 5, device=self.device, requires_grad=True)                 # state_dim
    dummy_action = torch.randn(1, 1, device=self.device, requires_grad=True)                # action_dim
    torch.onnx.export(
        actor_model, 
        dummy_state, 
        "DDPG_actor.onnx", 
        verbose=False,
        input_names= ['input']
        )
    torch.onnx.export(
        critic_model, 
        [dummy_state, dummy_action], 
        "DDPG_critic.onnx", 
        verbose=False
        )         
