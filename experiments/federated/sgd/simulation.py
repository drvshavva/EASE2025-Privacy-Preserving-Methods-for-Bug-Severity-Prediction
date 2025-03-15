from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from client import client_fn
from server import server_fn

# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)

# Create your ServerApp passing the server generation function
server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=3,
)
