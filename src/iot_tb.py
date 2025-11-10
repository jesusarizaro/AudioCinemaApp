#!/usr/bin/env python3
from typing import Dict, Any
import json, ssl
import paho.mqtt.client as mqtt

def send_json_to_thingsboard(payload: Dict[str,Any], host: str, port: int, token: str, use_tls: bool) -> bool:
    """
    Envía telemetry JSON a ThingsBoard:
    - host: thingsboard.cloud
    - port: 1883 (sin TLS) / 8883 (TLS)
    - token: access token del dispositivo
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(token)

    if use_tls:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        if port == 1883:
            port = 8883

    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
        topic = "v1/devices/me/telemetry"
        res = client.publish(topic, json.dumps(payload), qos=1)
        res.wait_for_publish(timeout=5.0)
        ok = res.is_published()
    except Exception:
        ok = False
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass
    return ok
