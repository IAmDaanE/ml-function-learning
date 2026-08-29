# ML Function Learning

Train an AI to take the shape of any mathmematical function.

## About the Project

This was written in python using [my very own neural network library](https://github.com/IAmDaanE/neural-network-library). The input into the neural network is just the normalized x value and the network should spit out the y value for that x value. The function's are visualized using MatPlotLib and the neural network is visualized by that neural network library using pygame.

## Getting Started

### Getting the Source

This project is [hosted on GitHub](https://github.com/IAmDaanE/ml-function-learning). You can download the zip or clone this project directly using this command:

```
git clone git@github.com:IAmDaanE/ml-function-learning.git
```

### Running the Program

Requirements: You must have Python 3.9 - 3.13.
1. Clone the repository or download the zip and unpack it to your directory of choice.
2. Navigate to that directory in a terminal.
3. In a venv or the global python version install the needed libraries.
    ```
    pip install -r requirements.txt
    ```
4. Run the program.
    ```
    python train.py
    ```
5. To change the function the AI should learn change this line in train.py:
    ```python
    correct_y = np.sin(x / 15) * 10 # the actual function to learn
    ```

## License

This project is open-source and available under the MIT License.