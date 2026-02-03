import json
import os
from datetime import datetime
from unittest.mock import MagicMock
import requests
import base64
import threading
import opentelemetry.proto.trace.v1.trace_pb2 as trace_pb2
from google.protobuf.json_format import MessageToDict
import atexit


class CustomHTTPInterceptor:
    def __init__(
        self,
        output_dir="./local_requests",
        intercept_keywords=None,
        otlp_keywords=None,
        enable_cache=True,
        response_status=200,
        timestamp_format="%Y%m%d_%H%M%S_%f",
    ):
        # Configuration parameters
        self.output_dir = output_dir
        self.enable_cache = enable_cache
        self.response_status = response_status
        self.timestamp_format = timestamp_format

        # Convert to set for better lookup performance
        self.intercept_keywords = set(
            intercept_keywords
            or [
                "agentops.ai",
                "smith.langchain",
                "api.coze.cn/v1/loop",
                "langfuse",
                "6006",
            ]
        )
        self.otlp_keywords = set(otlp_keywords or ["otlp.agentops.ai", "api.coze.cn/v1/loop", "langfuse", "6006"])

        # Internal state
        self._url_cache = {} if enable_cache else None
        self._id_fields = ["trace_id", "span_id", "parent_span_id"]
        self._pending_lock = threading.Lock()
        self._pending = 0
        self._pending_zero = threading.Event()
        self._pending_zero.set()

        # Pre-create mock response
        self._mock_response = self._create_mock_response()

        # Start interception
        self._setup_patches()
        atexit.register(self._flush_at_exit)

    def _should_intercept(self, url):
        url_str = str(url)
        if self._url_cache is not None and url_str in self._url_cache:
            return self._url_cache[url_str]

        url_lower = url_str.lower()
        result = any(k in url_lower for k in self.intercept_keywords)

        if self._url_cache is not None:
            self._url_cache[url_str] = result
        return result

    def _create_mock_response(self, url=None, method="GET"):
        mock_resp = MagicMock()
        mock_resp.status_code = self.response_status
        mock_resp.json.return_value = {"success": True, "local": True}
        mock_resp.raise_for_status = lambda: None
        return mock_resp

    def flatten_otlp_data(self, data):
        """Smart flatten OTLP data, preserving all information"""
        if isinstance(data, list):
            return [self.flatten_otlp_data(item) for item in data]

        if isinstance(data, dict):
            # Process attributes
            if "attributes" in data and isinstance(data["attributes"], list):
                attrs = {}
                for attr in data["attributes"]:
                    if isinstance(attr, dict) and "key" in attr:
                        key = attr["key"]
                        if "value" in attr:
                            value = self._extract_typed_value(attr["value"])
                            attrs[key] = value
                        else:
                            attrs[key] = attr
                data["attributes"] = attrs

            # Recursively process all nested structures without field restrictions
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = self.flatten_otlp_data(value)

        return data

    def _extract_typed_value(self, value_obj):
        """Extract typed value, handle all OTLP value types"""
        if not isinstance(value_obj, dict):
            return value_obj

        # OTLP value type mapping
        type_handlers = {
            "string_value": lambda x: str(x),
            "int_value": lambda x: int(x),
            "double_value": lambda x: float(x),
            "bool_value": lambda x: bool(x),
            "bytes_value": lambda x: base64.b64encode(x).decode() if isinstance(x, bytes) else str(x),
            "array_value": self._handle_array_value,
            "kvlist_value": self._handle_kvlist_value,
        }

        for type_key, handler in type_handlers.items():
            if type_key in value_obj:
                try:
                    return handler(value_obj[type_key])
                except Exception as e:
                    print(f"Value extraction error for {type_key}: {e}")
                    return value_obj[type_key]

        # If no matching type, return the entire object
        return value_obj

    def _handle_array_value(self, array_obj):
        """Handle array type value"""
        if isinstance(array_obj, dict) and "values" in array_obj:
            return [self._extract_typed_value(v) for v in array_obj["values"]]
        return array_obj

    def _handle_kvlist_value(self, kvlist_obj):
        """Handle key-value list type value"""
        if isinstance(kvlist_obj, dict) and "values" in kvlist_obj:
            result = {}
            for kv in kvlist_obj["values"]:
                if isinstance(kv, dict) and "key" in kv and "value" in kv:
                    result[kv["key"]] = self._extract_typed_value(kv["value"])
            return result
        return kvlist_obj

    def _parse_otlp_data(self, data: bytes):
        """Fully parse OTLP data, preserving original structure"""
        try:
            trace_data = trace_pb2.TracesData()
            trace_data.ParseFromString(data)
            result = MessageToDict(
                trace_data,
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
                use_integers_for_enums=False,
            )
            return self._process_converted_data(result)
        except Exception as e:
            print(f"OTLP parsing error: {e}")
            return None

    def _process_converted_data(self, data):
        """Process converted data"""
        if isinstance(data, dict):
            # Convert ID fields
            for field in self._id_fields:
                if field in data and data[field]:
                    try:
                        data[field] = "0x" + base64.b64decode(data[field]).hex()
                    except:
                        pass

            # Recursively process all fields
            for key, value in data.items():
                data[key] = self._process_converted_data(value)

        elif isinstance(data, list):
            data = [self._process_converted_data(item) for item in data]

        # Flatten processing
        return self.flatten_otlp_data(data)

    def _process_data(self, data, url):
        if data is None or isinstance(data, (dict, list, int, float, bool)):
            return data

        url_lower = str(url).lower()

        if isinstance(data, (bytes, bytearray)):
            if any(k in url_lower for k in self.otlp_keywords):
                try:
                    return self._parse_otlp_data(data)
                except:
                    pass

            try:
                text = data.decode("utf-8")
                try:
                    return json.loads(text)
                except:
                    return text
            except:
                return {"__binary__": True, "base64": base64.b64encode(data).decode()}

        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return data

        return data

    def _save_request(self, method, url, data=None):
        # Async save, do not block main thread
        with self._pending_lock:
            self._pending += 1
            self._pending_zero.clear()

        t = threading.Thread(target=self._save_request_sync_wrapper, args=(method, url, data), daemon=False)
        t.start()

    def _save_request_sync(self, method, url, data=None):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime(self.timestamp_format)
            processed_data = self._process_data(data, url)

            url_lower = str(url).lower()
            if (
                isinstance(processed_data, dict)
                and processed_data.get("__binary__")
                and "api.coze.cn/v1/loop" in url_lower
            ):
                try:
                    binary_data = base64.b64decode(processed_data["base64"])
                    processed_data = self._parse_otlp_data(binary_data)
                except Exception as e:
                    print(f"Failed to parse OTLP: {e}")

            req_data = {
                "timestamp": timestamp,
                "method": method,
                "url": str(url),
                "data": processed_data,
            }

            filepath = os.path.join(self.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(req_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Save error: {e}")

    def _save_request_sync_wrapper(self, method, url, data=None):
        try:
            self._save_request_sync(method, url, data)
        finally:
            with self._pending_lock:
                self._pending -= 1
                if self._pending == 0:
                    self._pending_zero.set()

    def _flush_at_exit(self):
        self._pending_zero.wait()

    def _setup_patches(self):
        original_request = requests.request
        original_session_request = requests.Session.request

        def patched_request(method, url, **kwargs):
            if self._should_intercept(url):
                data = kwargs.get("json") or kwargs.get("data")
                self._save_request(method, url, data)
                return self._create_mock_response(url, method)
            return original_request(method, url, **kwargs)

        def patched_session_request(session_self, method, url, **kwargs):
            if self._should_intercept(url):
                data = kwargs.get("json") or kwargs.get("data")
                self._save_request(method, url, data)
                return self._create_mock_response(url, method)
            return original_session_request(session_self, method, url, **kwargs)

        requests.request = patched_request
        requests.Session.request = patched_session_request
