#!/usr/bin/env python3
import json, os
import paho.mqtt.client as mqtt

def send_json_to_thingsboard(payload: dict, host: str, port: int, token: str, use_tls: bool=False):
    """Publica JSON en ThingsBoard."""
    try:
        client_id = f"AudioCinemaPi-{os.uname().nodename}-{os.getpid()}"
        client = mqtt.Client(client_id=client_id, clean_session=True)
        client.username_pw_set(token)
        if use_tls:
            import ssl
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
        client.connect(host, port, keepalive=30)
        topic = "v1/devices/me/telemetry"
        result, _ = client.publish(topic, json.dumps(payload), qos=1)
        client.loop(timeout=2.0)
        client.disconnect()
        return result == mqtt.MQTT_ERR_SUCCESS
    except Exception as e:
        print(f"Error MQTT: {e}")
        return False
