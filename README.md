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
To start a run with a different config execute:
```
python3 train.py -config=/path/to/config.py
```

To resume a run:
```
python3 train.py --config.run_id="$run_id" --config.resume="must"
```

# Testing
Tests are included in parallel to the implementation of most components.
To run the tests install all dependencies and execute
```
python3 -m pip install pytest
```
next you can execute all tests via
```
python3 -m pytest
```
individual files are functions can be tested in the following ways:
```
python3 -m pytest /path/to/test_file.py
python3 -m pytest /path/to/test_file.py::test_function
```
