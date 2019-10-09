import network_stability

net = network_stability.NetworkTest()
net.connection_test_interval(hours=1)
net.export_connection_results('connection.csv')
net.report_connection('connection.png')
net.speed_test_interval(hours=1)
net.export_speed_results('speed.csv')
net.report_speed('speed.png')
print('Done!')
