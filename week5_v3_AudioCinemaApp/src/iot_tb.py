#!/usr/bin/env python3
from __future__ import annotations
import json
import paho.mqtt.client as mqtt

def send_json_to_thingsboard(payload: dict, host: str, port: int, token: str, use_tls: bool) -> bool:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(token)
        if use_tls:
            import ssl
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
        client.connect(host, int(port), keepalive=30)
        client.publish("v1/devices/me/attributes", json.dumps({"pi_online": True}), qos=1)
        r = client.publish("v1/devices/me/telemetry", json.dumps(payload), qos=1)
        client.loop(timeout=2.0)
        client.disconnect()
        return r.rc == mqtt.MQTT_ERR_SUCCESS
    except Exception:
        return False
