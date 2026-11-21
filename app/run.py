"""Run SORT locally; --https includes a private development CA for your phone."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
from pathlib import Path
import socket
import sys
from threading import Thread


APP_DIR = Path(__file__).resolve().parent


def local_addresses() -> list[str]:
    """Find IPv4 LAN addresses without contacting an external service."""
    addresses = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not ipaddress.ip_address(address).is_loopback:
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def make_certificates(directory: Path, addresses: list[str]) -> tuple[Path, Path]:
    """Keep the same root CA between runs; renew the leaf with current LAN IPs."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory.mkdir(parents=True, exist_ok=True)
    ca_path = directory / "local-ca.crt"
    ca_key_path = directory / "local-ca.key"
    now = datetime.now(timezone.utc)
    if ca_path.exists() and ca_key_path.exists():
        ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        if ca.public_key().public_numbers() != ca_key.public_key().public_numbers():
            raise ValueError("The local CA and its key do not match.")
        if ca.not_valid_after_utc <= now + timedelta(days=1):
            raise ValueError("The local CA has expired. Remove app/.certs and trust a newly generated CA.")
    elif ca_path.exists() or ca_key_path.exists():
        raise ValueError("Incomplete app/.certs CA. Restore both local-ca.crt and local-ca.key, or recreate them.")
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SORT local development CA")])
        ca = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=None, decipher_only=None,
            ), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        ca_key_path.write_bytes(ca_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sans = []
    for address in sorted(set(["localhost", "127.0.0.1", *addresses])):
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(address)))
        except ValueError:
            sans.append(x509.DNSName(address))
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SORT local server")]))
        .issuer_name(ca.subject).public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(min(now + timedelta(days=365), ca.not_valid_after_utc))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(
            digital_signature=True, content_commitment=False, key_encipherment=True,
            data_encipherment=False, key_agreement=False, key_cert_sign=False,
            crl_sign=False, encipher_only=None, decipher_only=None,
        ), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path = directory / "server.crt"
    key_path = directory / "server.key"
    cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(server_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    return cert_path, key_path


def certificate_server(host: str, port: int, ca_path: Path) -> ThreadingHTTPServer:
    """Serve only the public certificate and setup text, never private keys."""
    class SetupHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/local-ca.crt":
                body = ca_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-x509-ca-cert")
                self.send_header("Content-Disposition", 'attachment; filename="sort-local-ca.crt"')
            elif self.path == "/":
                body = (
                    '<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    '<title>SORT camera setup</title><body style="font:18px system-ui;'
                    'max-width:35rem;margin:3rem auto;padding:1.5rem;line-height:1.6">'
                    '<h1>Enable the phone camera</h1><p>For this local development server, '
                    'download the public certificate and trust it on your phone.</p>'
                    '<p><a href="/local-ca.crt">Download SORT certificate</a></p>'
                    '<p>iPhone: install the downloaded profile in Settings, General, '
                    'VPN &amp; Device Management. Then enable SORT under General, About, '
                    'Certificate Trust Settings.</p><p>Android: use Settings to install '
                    'the downloaded certificate as a CA certificate. Menu names vary by phone.</p>'
                    '<p>Return to the HTTPS address printed in your terminal. '
                    'Only install this certificate on your own development device. '
                    'Remove it when you no longer use SORT.</p></body></html>'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            else:
                self.send_error(404)
                return
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer((host, port), SetupHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: all local interfaces)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https", action="store_true", help="Generate a local CA and serve HTTPS")
    parser.add_argument("--certfile", type=Path, help="Use an existing trusted TLS certificate")
    parser.add_argument("--keyfile", type=Path, help="Private key for --certfile")
    args = parser.parse_args()
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("--certfile and --keyfile must be supplied together")
    if args.https and args.certfile:
        parser.error("Use --https or your own certificate, not both")
    if not 1 <= args.port <= 65534:
        parser.error("--port must be between 1 and 65534")

    try:
        import uvicorn
    except ImportError:
        parser.exit(1, "Install dependencies first: python -m pip install -r app/requirements.txt\n")
    addresses = local_addresses()
    certfile, keyfile = args.certfile, args.keyfile
    setup_server = None
    try:
        if args.https:
            certificate_addresses = addresses + ([] if args.host == "0.0.0.0" else [args.host])
            certfile, keyfile = make_certificates(APP_DIR / ".certs", certificate_addresses)
            setup_server = certificate_server(args.host, args.port + 1, APP_DIR / ".certs/local-ca.crt")
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"HTTPS setup failed: {error}\nSee app/README.md for certificate setup.\n")

    scheme = "https" if certfile else "http"
    print(f"\nSORT: {scheme}://localhost:{args.port}", flush=True)
    phone_addresses = addresses if args.host == "0.0.0.0" else [args.host]
    for address in phone_addresses:
        print(f"Phone (same Wi-Fi): {scheme}://{address}:{args.port}", flush=True)
        if setup_server:
            print(f"First trust the phone certificate: http://{address}:{args.port + 1}", flush=True)
    if not certfile:
        print("Live camera works on localhost. For your phone, restart with --https.", flush=True)
    elif setup_server:
        print("A browser certificate warning means you still need to trust the local CA.", flush=True)
    print("Keep the computer running. Use a trusted local network; this app has no sign-in.\n", flush=True)
    sys.path.insert(0, str(APP_DIR.parent))
    try:
        uvicorn.run(
            "app.server:app", host=args.host, port=args.port,
            ssl_certfile=str(certfile) if certfile else None,
            ssl_keyfile=str(keyfile) if keyfile else None,
            access_log=False,
        )
    finally:
        if setup_server:
            setup_server.shutdown()
            setup_server.server_close()


if __name__ == "__main__":
    main()
