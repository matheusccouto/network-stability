# Network Stability
Python module for network connectivity and speed test.

### Prerequisites

```
Python 3.6+
```

### Installing

Use the package manager pip to install the package.
```bash
pip install network_stability
```

## Usage

```python
import network_stability

net = network_stability.NetworkTest()
# Run connectivity test.
net.connection_test_interval(hours=0.5)
net.export_connection_results('connection.csv')
net.report_connection('connection.png')
# Run speed test.
net.speed_test_interval(minutes=30)
net.export_speed_results('speed.csv')
net.report_speed('speed.png')
```

## Author

* **Matheus Couto** - [Linkedin](https://www.linkedin.com/in/matheusccouto/)

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Acknowledgments

* This project inspiration came after struggles to watch The Office in my TV. I had  a  wifi  extender that I was 
distrusting. I needed some tests to check if it was messing with my network stability.
