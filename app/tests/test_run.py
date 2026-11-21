"""Verify the phone TLS setup without installing certificates on the host."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from app.run import certificate_server, make_certificates


class PhoneSetupTests(unittest.TestCase):
    def test_certificate_passes_strict_tls_validation(self):
        class ProbeHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"trusted")

            def log_message(self, *_args):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            certificate, key = make_certificates(directory, ["127.0.0.1"])
            tls_server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls_server.load_cert_chain(certificate, key)
            server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
            server.socket = tls_server.wrap_socket(server.socket, server_side=True)
            Thread(target=server.serve_forever, daemon=True).start()
            try:
                trusted_client = ssl.create_default_context(cafile=directory / "local-ca.crt")
                with urlopen(f"https://127.0.0.1:{server.server_port}/", context=trusted_client, timeout=5) as response:
                    self.assertEqual(response.read(), b"trusted")
            finally:
                server.shutdown()
                server.server_close()

    def test_certificate_covers_lan_address_and_preserves_trusted_ca(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            certificate, _key = make_certificates(directory, ["192.168.1.42", "localhost"])
            ca_bytes = (directory / "local-ca.crt").read_bytes()
            ca = x509.load_pem_x509_certificate(ca_bytes)
            leaf = x509.load_pem_x509_certificate(certificate.read_bytes())
            leaf.verify_directly_issued_by(ca)
            sans = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            self.assertIn("localhost", sans.get_values_for_type(x509.DNSName))
            self.assertIn("192.168.1.42", [str(item) for item in sans.get_values_for_type(x509.IPAddress)])
            usages = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
            self.assertIn(ExtendedKeyUsageOID.SERVER_AUTH, usages)
            make_certificates(directory, ["192.168.1.99"])
            self.assertEqual(ca_bytes, (directory / "local-ca.crt").read_bytes())
            renewed = x509.load_pem_x509_certificate(certificate.read_bytes())
            renewed.verify_directly_issued_by(ca)
            updated_sans = renewed.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            self.assertIn("192.168.1.99", [str(item) for item in updated_sans.get_values_for_type(x509.IPAddress)])

    def test_setup_server_serves_public_certificate_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            make_certificates(directory, ["127.0.0.1"])
            server = certificate_server("127.0.0.1", 0, directory / "local-ca.crt")
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/local-ca.crt", timeout=5) as response:
                    self.assertEqual(response.read(), (directory / "local-ca.crt").read_bytes())
                for path in ["/local-ca.key", "/server.key", "/../local-ca.key"]:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + path, timeout=5)
                    self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
