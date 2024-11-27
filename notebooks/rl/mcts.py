import encoder
import math
from threading import Thread
import time


def calc_uct(edge, parent_visits):
    """
    Calculate the Upper Confidence Bound for Trees (UCT) value for a given edge.
    """
    uct_value = edge.get_q() + (edge.get_probability() * 1.5 *
                                math.sqrt(parent_visits) / (1 + edge.get_n()))
    return uct_value


class TreeNode:
    def __init__(self, chessboard, win_probability, move_probabilities):
        self.visits, self.cumulative_q, self.outgoing_edges = 1.0, win_probability, []
        for index, legal_move in enumerate(chessboard.legal_moves):
            self.outgoing_edges.append(Edge(legal_move, move_probabilities[index]))

    def get_n(self):
        return self.visits

    def get_q(self):
        return self.cumulative_q / self.visits

    def select_best_edge(self):
        """
        Select the edge with the highest UCT value.
        """
        best_uct_value, selected_edge = float('-inf'), None
        for current_edge in self.outgoing_edges:
            uct_score = calc_uct(current_edge, self.visits)
            if uct_score > best_uct_value:
                best_uct_value = uct_score
                selected_edge = current_edge
        return selected_edge

    def select_max_n(self):
        """
        Select the edge with the highest visit count (N).
        """
        highest_visits, best_edge = -1, None
        for edge in self.outgoing_edges:
            visit_count = edge.get_n()
            if highest_visits < visit_count:
                highest_visits, best_edge = visit_count, edge
        return best_edge


class Edge:
    def __init__(self, chess_move, move_prob):
        self.move, self.probability, self.child, self.virtual_losses = chess_move, move_prob, None, 0.

    def has_child(self):
        return self.child is not None

    def get_n(self):
        """
        Get the visit count for this edge, including virtual losses.
        """
        if self.has_child():
            return self.child.visits + self.virtual_losses
        return self.virtual_losses

    def get_q(self):
        """
        Get the Q value (expected win rate) for this edge.
        """
        if self.has_child():
            total_q = self.child.cumulative_q + self.virtual_losses
            total_visits = self.child.visits + self.virtual_losses
            return 1.0 - (total_q / total_visits)
        return 0.0

    def get_probability(self):
        return self.probability

    def expand(self, chessboard, new_q, move_probabilities):
        """
        Expand this edge to create a child node.
        """
        if self.child is None:
            self.child = TreeNode(chessboard, new_q, move_probabilities)
            return True
        return False

    def get_child(self):
        return self.child

    def get_move(self):
        return self.move

    def add_virtual_loss(self):
        self.virtual_losses += 1

    def clear_virtual_loss(self):
        self.virtual_losses = 0.


class Root(TreeNode):
    def __init__(self, chessboard, neural_net):
        root_value, move_probs = encoder.call_neural_net(chessboard, neural_net)
        initial_Q = root_value / 2. + 0.5
        super().__init__(chessboard, initial_Q, move_probs)
        self.same_paths = 0

    def select_task(self, chessboard, node_trace, edge_trace):
        """
        Select a task (path in the tree) for exploration during rollouts.
        """
        current_node = self
        while True:
            node_trace.append(current_node)
            selected_edge = current_node.select_best_edge()
            edge_trace.append(selected_edge)
            if selected_edge == None:
                break
            selected_edge.add_virtual_loss()
            chessboard.push(selected_edge.get_move())
            if not selected_edge.has_child():
                break
            current_node = selected_edge.get_child()

    def parallel_rollouts(self, chessboard, neural_net, num_parallel_rollouts):
        cloned_boards = [chessboard.copy() for _ in range(num_parallel_rollouts)]
        node_traces = [[] for _ in range(num_parallel_rollouts)]
        edge_traces = [[] for _ in range(num_parallel_rollouts)]
        threads = []

        for i in range(num_parallel_rollouts):
            thread = Thread(
                target=self.select_task,
                args=(cloned_boards[i], node_traces[i], edge_traces[i])
            )
            threads.append(thread)
            thread.start()
            time.sleep(0.0001) # Minor stagger to prevent contention

        for thread in threads:
            thread.join()

        values, move_prob_batches = encoder.call_batched_neural_net(cloned_boards, neural_net)

        for i in range(num_parallel_rollouts):
            chessboard, value = cloned_boards[i], values[i]
            edge = edge_traces[i][-1] if edge_traces[i] else None

            if edge:
                new_q = value * 0.5 + 0.5
                is_unexpanded = edge.expand(chessboard, new_q, move_prob_batches[i])
                if not is_unexpanded:
                    self.same_paths += 1
                new_q = 1.0 - new_q
            else:
                winner = encoder.game_result(chessboard.result())
                if not chessboard.turn:
                    winner *= -1
                new_q = winner * 0.5 + 0.5

            self._update_nodes_and_edges(node_traces[i], edge_traces[i], new_q)

    def _update_nodes_and_edges(self, node_trace, edge_trace, new_q):
        """Helper function to update nodes and edges after a rollout."""
        for r, node in enumerate(reversed(node_trace)):
            node.visits += 1
            is_even_depth = (len(node_trace) - 1 - r) % 2 == 0
            node.cumulative_q += new_q if is_even_depth else 1.0 - new_q

        for edge in edge_trace:
            if edge:
                edge.clear_virtual_loss()
