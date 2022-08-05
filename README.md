# little_vision
experiments with deep neural networks


# Training
Working example of training runs can be found as
iPython notebooks that may be run in Google Colaboratory.

# Warning!
Training  using the ```train.py``` script is sadly not working.
To fix this crisis traing notebooks (as described above) were implemented.

# Example of how it should have been: 
To run the MNIST example with Weights and Biases logging:
```
python3 train.py
```
To resume a run:
```
python3 train.py --config.run_id="$run_id" --config.resume="must"
```