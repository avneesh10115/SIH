import 'package:flutter/material.dart';
import 'package:location/location.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: SOSPage(),
    );
  }
}

class SOSPage extends StatefulWidget {
  @override
  _SOSPageState createState() => _SOSPageState();
}

class _SOSPageState extends State<SOSPage> {
  Location location = new Location();
  late bool _serviceEnabled;
  late PermissionStatus _permissionGranted;
  LocationData? _locationData;
  String? _responseMessage;
  double dangerLevel = 70; // Default value, will be updated from server
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _getLocation();
    _startAutoRequest();
  }

  Future<void> _getLocation() async {
    _serviceEnabled = await location.serviceEnabled();
    if (!_serviceEnabled) {
      _serviceEnabled = await location.requestService();
      if (!_serviceEnabled) {
        return;
      }
    }

    _permissionGranted = await location.hasPermission();
    if (_permissionGranted == PermissionStatus.denied) {
      _permissionGranted = await location.requestPermission();
      if (_permissionGranted != PermissionStatus.granted) {
        return;
      }
    }

    _locationData = await location.getLocation();
    setState(() {});

    if (_locationData != null) {
      _sendLocationToServer(_locationData!.latitude, _locationData!.longitude);
    }
  }

  Future<void> _sendLocationToServer(
      double? latitude, double? longitude) async {
    final url = Uri.parse(
        'http://10.0.2.2:5000/check'); // Replace with your server address
    final response = await http.post(
      url,
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, dynamic>{
        'lat': latitude,
        'long': longitude,
      }),
    );

    if (response.statusCode == 200) {
      final Map<String, dynamic> responseData = jsonDecode(response.body);
      setState(() {
        _responseMessage = responseData['CCTV'];
      });
    } else {
      setState(() {
        _responseMessage = 'Failed to send location data';
        dangerLevel = 0; // Default value on failure
      });
    }
  }

  Future<void> getSos() async {
    final url = Uri.parse('http://10.0.2.2:5000/getsos');
    final resp = await http.get(
      url,
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
    );
    if (resp.statusCode == 200) {
      setState(() {
        dangerLevel = double.parse(resp.body) * 100;
      });
    } else {
      setState(() {
        _responseMessage = 'SOS Request Failed';
      });
    }
  }

  void _startAutoRequest() {
    _timer = Timer.periodic(Duration(seconds: 2), (timer) async {
      getSos();
    });
  }

  @override
  void dispose() {
    _timer?.cancel(); // Cancel the timer when the widget is disposed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background image
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage(
                    "assets/space_background.jpg"), // Ensure this file is in your assets
                fit: BoxFit.cover,
              ),
            ),
          ),
          // Centered content
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    _getLocation();
                  },
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 20, horizontal: 40),
                    backgroundColor: Colors.red,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30.0),
                    ),
                  ),
                  child: Text(
                    'SOS',
                    style: TextStyle(
                      fontSize: 40,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                SizedBox(height: 20),
                Text(
                  'Danger Level',
                  style: TextStyle(
                    fontSize: 20,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 10),
                SizedBox(
                  width: 200,
                  height: 20,
                  child: LinearProgressIndicator(
                    value: dangerLevel / 100,
                    backgroundColor: Colors.grey[300],
                    color: Colors.red,
                    minHeight: 20,
                  ),
                ),
                SizedBox(height: 20),
                if (_responseMessage != null)
                  Text(
                    _responseMessage!,
                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
