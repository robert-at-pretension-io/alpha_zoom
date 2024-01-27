// fn main() {
// neural network arch
// Random initialization of weights

// MCTS
// Battle the current neural network against the best neural network
// Sample from the probability distribution over the actions for a particular state
// Build tree, use NN to evaluate nodes
//

// If the new network wins >= 55% of games, use this new network for self play to create training data for the neural network. Otherwise, use self play of the previous network to create training data for the neural network.

// Loss function to train the neural network

// }

extern crate tch;
use tch::nn;
use tch::Device;
use tch::Tensor;
use tch::nn::Module;
extern crate baz_core;
use baz_core::{Board, Height, Color};
use tch::Kind;

    pub fn to_input_tensor(board : Board) -> Tensor {
        let mut tensor = [[[0u8; 8]; 8]; 8];

        for piece in board.pieces.clone().iter() {
            let x = piece.position.x() as usize;
            let y = piece.position.y() as usize;

            // Encode presence
            tensor[0][y][x] = 1;

            // Encode color
            tensor[1][y][x] = match piece.color {
                Color::White => 1,
                Color::Black => 0,
            };

            // Encode height
            match piece.height {
                Height::One => {
                    tensor[2][y][x] = 1;
                }
                Height::Two => {
                    tensor[3][y][x] = 1;
                }
                Height::Three => {
                    tensor[4][y][x] = 1;
                }
                Height::Dead => {}
            };
        }

        // Encode scores and other game states in remaining slices...
        // For example, using slices 5 and 6 for white and black scores
        for y in 0..8 {
            for x in 0..8 {
                tensor[5][y][x] = board.white_score;
                tensor[6][y][x] = board.black_score;
            }
        }

        // Encode additional information in slice 7 if necessary...

        let flat: Vec<u8> = tensor.iter().flatten().flatten().copied().collect();

        let actual_tensor = Tensor::f_from_slice(&flat)
        .expect("Failed to create tensor from slice")
        .to_kind(Kind::Float) // Convert to Float Tensor
        .view([8, 8, 8]); // Reshape the tensor
        
        actual_tensor
    }

// A neural network with a shared convolutional base and two heads: policy and value.
#[derive(Debug)]
struct DualHeadedConvNet {
    shared_conv_layer: nn::Conv2D,
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl DualHeadedConvNet {
    fn new(vs: &nn::Path) -> DualHeadedConvNet {
        // Define the shared convolutional layer.
        let shared_conv_layer = nn::conv2d(
            vs,
            8 /* in_channels */,
            32 /* out_channels */,
            3 /* ks */,
            nn::ConvConfig::default()
        );

        // Define the policy head.
        // Assuming a game with at most 10 possible moves from any state.
        let policy_head = nn::linear(vs, 32 * 6 * 6, 10 /* num_moves */, Default::default());

        // Define the value head.
        let value_head = nn::linear(vs, 32 * 6 * 6, 1 /* output_value_size */, Default::default());

        DualHeadedConvNet {
            shared_conv_layer,
            policy_head,
            value_head,
        }
    }
}

// The implementation of nn::Module requires defining the forward method.
impl nn::Module for DualHeadedConvNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let conv_output = xs.apply(&self.shared_conv_layer);
        println!("Size after conv layer: {:?}", conv_output.size());

        // Replace 32 * 8 * 8 with the correct dimensions based on the output size of shared_conv_layer

        let correct_dimensions = 1152;

        let xs_reshaped = conv_output.view([-1, correct_dimensions]);

        let policy_logits = xs_reshaped.apply(&self.policy_head);

        println!("Size after policy head: {:?}", policy_logits.size());

        let value_pred = xs_reshaped.apply(&self.value_head).sigmoid();

        println!("Size after value head: {:?}", value_pred.size());

        // Since value_pred is 1D, we need to add a dimension to it before concatenation
        let value_pred_reshaped = value_pred.unsqueeze(0);

// Reshape policy_logits to match the dimensions for concatenation
let policy_logits_reshaped = policy_logits.view([-1, 1, 10]); // Now it is [1, 1, 10]

// Concatenate policy_logits_reshaped and value_pred_reshaped along the last dimension
Tensor::cat(&[policy_logits_reshaped, value_pred_reshaped], 2)

    }
}

fn compute_loss(
    predictions: (Tensor, Tensor),
    true_outcomes: Tensor,
    policy_targets: Tensor,
    legal_moves_mask: Tensor
) -> Tensor {
    let (policy_logits, value_preds) = predictions;

    // Value loss - mean squared error.
    let value_loss = (&value_preds - &true_outcomes).pow(&Tensor::from(2)).mean(Kind::Float);

    // Policy loss - masked cross-entropy loss.
    let policy_probs = (&policy_logits * &legal_moves_mask).softmax(-1, Kind::Float); // Mask and softmax
    let policy_loss = -(&policy_targets * policy_probs.log()).sum(Kind::Float);

    // Combined loss
    let total_loss = value_loss + policy_loss;
    total_loss
}

use crate::tch::IndexOp;

fn main() {
    // Use the CUDA device if available, otherwise CPU
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    // Instantiate the dual-headed neural network
    let net = DualHeadedConvNet::new(&vs.root());

    // Create a dummy input tensor representing the board state
    let input = Tensor::randn(&[1, 8, 8, 8], (tch::Kind::Float, device));

    println!("Input (boardgame shape) is: {:?}", input.size());

    // Forward pass to get the policy and value
    // let (policy, value) = net.forward(&input);


    let result = net.forward(&input);


// Extract policy_logits: take the first 10 elements of the last dimension
let policy_logits = result.i((.., .., 0..10)).squeeze();

// Extract value_pred: take the last element of the last dimension
let value_pred = result.i((.., .., 10)).squeeze();

    // If you need to convert value_pred to a scalar float
    let _value_pred_scalar = value_pred.double_value(&[]);

    // Print out the size of the policy and value outputs
    println!("Policy vector size: {:?}", policy_logits.size());
    println!("Value size: {:?}", value_pred.size());

    // Create some example data that would come from your model and MCTS search
    let value_predictions = Tensor::from(0.5); // Dummy predicted value
    let true_outcomes = Tensor::from(1.0); // Dummy true outcome (e.g., win)
    let policy_logits = Tensor::randn(&[1, 10], (tch::Kind::Float, device)); // Dummy logits for 10 possible actions
    let policy_targets = Tensor::rand(&[1, 10], (tch::Kind::Float, device)); // Dummy MCTS policy vector

    let policy_logits_clone = policy_logits.shallow_clone();
    let loss = compute_loss(
        (value_predictions, policy_logits),
        true_outcomes,
        policy_targets,
        policy_logits_clone
    );
    
    // Print the loss
    println!("{:?}", loss);
}

// Boom and Zoom Setup and Rules:

// The game has the following configuration:
// The game is played on an 8 × 8 board, with 12 stackable black counters and 12 stackable white counters. Initially, each side places four stacks of height 3 on the four central squares of the home row:

// The rules are simple: each player has four towers, three pieces high, that can "boom" (fire) or "zoom" (move) a number of spaces equal to the tower's height. But with those deceptively simple ingredients, veteran designer Ty Bomba has created his masterpiece, the game that he himself acknowledges as his own best design. The trick is that the game ends when only one player's pieces remain on the board, and the player who managed to exit the most pieces off of the opponent's side of the map wins. Because of this, players can't concentrate on just blocking/attacking or just advancing, but must strike a difficult and subtle balance between the two.

// Boardgame State Representation

// To represent the board state for a neural network in the game "Boom and Zoom," we can encode the board as multi-channel 2D grids. This is similar to how the state is represented in games like chess or Go when using neural networks. Each channel will represent a specific piece of information about the game state.

// Given the game's configuration and rules, here's how we might design the state representation:

// 1. **Stack Height**: Since each tower can have a height from 1 to 3 pieces, we would need channels to represent the height of the stacks for each player. This can be encoded using three channels for each player, where each channel contains a binary encoding of whether a cell contains a tower of a specific height. For example:

//    - Channel 1 (Black): 1 if a cell contains a black stack of height 1, otherwise 0.
//    - Channel 2 (Black): 1 if a cell contains a black stack of height 2, otherwise 0.
//    - Channel 3 (Black): 1 if a cell contains a black stack of height 3, otherwise 0.
//    - Channel 4 (White): 1 if a cell contains a white stack of height 1, otherwise 0.
//    - Channel 5 (White): 1 if a cell contains a white stack of height 2, otherwise 0.
//    - Channel 6 (White): 1 if a cell contains a white stack of height 3, otherwise 0.

// 2. **Current Player**: You can add an additional channel to indicate the current player. This channel would be filled with 1s if it is black's turn to move and 0s for white (or vice versa).

// 3. **Exited Pieces Counter**: Since the objective includes exiting the most pieces off of the opponent’s side, we should track the number of exited pieces for each player. This can be encoded using two channels, one for each player, with each cell always containing the current count of the player's exited pieces.

//     - Channel 7 (Black Exited Pieces): This would be a single value repeated across the entire channel indicating how many black pieces have exited.
//     - Channel 8 (White Exited Pieces): Similarly, this channel would have a single value indicating the count of exited white pieces.

// With this configuration, the neural network's input layer would expect an 8 x 8 x 8 tensor. Each 8 x 8 slice of this tensor corresponds to one of the channels described above.

// Here's a visual example of what the channels for stack heights might look like for a simple 2x2 board, where '0' represents an empty cell, and '1' represents the presence of a stack of a specific height:

// Output representation (policy vector)

// In many games, including "Boom and Zoom," the number of legal moves can vary depending on the game state. Neural networks, however, typically require a fixed-size input and output. To handle this, you'll often need to represent the policy vector in a way that accounts for all possible moves in any state of the game.

// Here's how you can handle the variable number of possible moves:

// 1. **Fixed-Length Policy Vector**: Define the policy vector length to be equal to the maximum number of possible moves from any state. For each game state, you'll fill this vector with probabilities for all legal moves and zeros for all illegal ones.

// 2. **Masking Illegal Moves**: During the training phase (and possibly during gameplay), apply a mask to the policy vector. This mask will have zeros for positions corresponding to illegal moves and ones for legal moves. This way, when calculating the cross-entropy loss, the contribution of illegal moves will be zero.

// 3. **Normalization**: After masking, re-normalize the probabilities so that they sum to 1 across all legal moves.

// Let's adapt the function for calculating the policy loss, including handling the masking:
