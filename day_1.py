from typing import TypedDict

class State (TypedDict):
    graph_state: str


def node (state: State):
    graph_state = state['graph_state']
    print ("This is the Node")
    print (f"The graph State is {graph_state}")
    return {'graph_state': 'This is an update'}

if __name__ == '__main__':
    state = {'graph_state': 'Initial value '}
    new_state = node(state)
    print("Returned state:", new_state)