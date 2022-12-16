import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim) -> None:
        """DQN Network

        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
    
        super(QNetwork, self).__init__()
        
        # self.norm = torch.nn.LayerNorm(input_dim)

        self.Aslayer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Aslayer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Aslayer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Aslayer4 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Asfinal = torch.nn.Linear(hidden_dim, output_dim)

        self.Vlayer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Vlayer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Vlayer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Vlayer4 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.Vfinal = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value

        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)

        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)            
        """
        As = self.Aslayer1(x)
        As = self.Aslayer2(As)
        As = self.Aslayer3(As)
        As = self.Aslayer4(As)
        As = self.Asfinal(As)

        V = self.Vlayer1(x)
        V = self.Vlayer2(V)
        V = self.Vlayer3(V)
        V = self.Vlayer4(V)
        V = self.Vfinal(V)

        Q = V + (As - As.mean(axis=1).reshape(-1, 1))

        return Q
     
