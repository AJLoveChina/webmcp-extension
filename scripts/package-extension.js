import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const releasesDir = path.join(rootDir, 'releases');
const pemPath = path.join(rootDir, 'webmcp-extension.pem');

if (!fs.existsSync(distDir)) {
  throw new Error('dist/ directory does not exist. Run bundle and copy first.');
}

fs.mkdirSync(releasesDir, { recursive: true });

// Read version from manifest.json (fallback to package.json)
const manifest = JSON.parse(fs.readFileSync(path.join(rootDir, 'manifest.json'), 'utf8'));
const pkg = JSON.parse(fs.readFileSync(path.join(rootDir, 'package.json'), 'utf8'));
const version = manifest.version || pkg.version || '1.0.0';

const now = new Date();
const pad = (n) => String(n).padStart(2, '0');
const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
const baseName = `webmcp-extension-v${version}-${timestamp}`;

const zipPath = path.join(releasesDir, `${baseName}.zip`);
const crxPath = path.join(releasesDir, `${baseName}.crx`);

// 1. Build zip under releases/ from dist/
if (fs.existsSync(zipPath)) {
  fs.unlinkSync(zipPath);
}

execFileSync('zip', ['-r', '-q', zipPath, '.'], {
  cwd: distDir,
  stdio: 'inherit'
});

const zipBuffer = fs.readFileSync(zipPath);
console.log(`✓ Created releases/${path.basename(zipPath)} (${(zipBuffer.length / 1024).toFixed(1)} KB)`);

// 2. Build CRX3 package (webmcp-extension.crx) signed with an RSA-2048 key
let privateKeyPem;
if (fs.existsSync(pemPath)) {
  privateKeyPem = fs.readFileSync(pemPath, 'utf8');
} else {
  const { privateKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
  });
  privateKeyPem = privateKey;
  fs.writeFileSync(pemPath, privateKeyPem, { mode: 0o600 });
  console.log(`✓ Generated signing key ${path.basename(pemPath)}`);
}

const privateKeyObj = crypto.createPrivateKey(privateKeyPem);
const publicKeyDer = crypto
  .createPublicKey(privateKeyObj)
  .export({ type: 'spki', format: 'der' });

// Protobuf encoding helpers for CRX3 header
function encodeVarint(num) {
  const bytes = [];
  let n = num >>> 0;
  while (n >= 0x80) {
    bytes.push((n & 0x7f) | 0x80);
    n >>>= 7;
  }
  bytes.push(n);
  return Buffer.from(bytes);
}

function encodeLengthDelimitedField(fieldNumber, payloadBuffer) {
  const tag = (fieldNumber << 3) | 2;
  return Buffer.concat([
    encodeVarint(tag),
    encodeVarint(payloadBuffer.length),
    payloadBuffer
  ]);
}

// CrxFileHeader.SignedData has field 1 (crx_id: 16 bytes = first 16 bytes of SHA-256(publicKeyDer))
const pubKeyHash = crypto.createHash('sha256').update(publicKeyDer).digest();
const crxId = pubKeyHash.subarray(0, 16);
const signedHeaderData = encodeLengthDelimitedField(1, crxId);

// CRX3 signature is computed over:
// "CRX3 SignedData\x00" + uint32LE(signedHeaderData.length) + signedHeaderData + zipBuffer
const signaturePrefix = Buffer.from('CRX3 SignedData\x00', 'utf8');
const signedDataLenBuf = Buffer.alloc(4);
signedDataLenBuf.writeUInt32LE(signedHeaderData.length, 0);

const signer = crypto.createSign('RSA-SHA256');
signer.update(signaturePrefix);
signer.update(signedDataLenBuf);
signer.update(signedHeaderData);
signer.update(zipBuffer);
const signature = signer.sign(privateKeyObj);

// AsymmetricKeyProof: field 1 = public_key, field 2 = signature
const asymmetricKeyProof = Buffer.concat([
  encodeLengthDelimitedField(1, publicKeyDer),
  encodeLengthDelimitedField(2, signature)
]);

// CrxFileHeader: field 2 = sha256_with_rsa (AsymmetricKeyProof), field 10000 = signed_header_data
const crxFileHeader = Buffer.concat([
  encodeLengthDelimitedField(2, asymmetricKeyProof),
  encodeLengthDelimitedField(10000, signedHeaderData)
]);

// Final CRX3 binary layout: "Cr24" + uint32LE(3) + uint32LE(header.length) + header + zipBuffer
const magic = Buffer.from('Cr24', 'utf8');
const versionBuf = Buffer.alloc(4);
versionBuf.writeUInt32LE(3, 0);
const headerLenBuf = Buffer.alloc(4);
headerLenBuf.writeUInt32LE(crxFileHeader.length, 0);

const crxBuffer = Buffer.concat([
  magic,
  versionBuf,
  headerLenBuf,
  crxFileHeader,
  zipBuffer
]);

fs.writeFileSync(crxPath, crxBuffer);
console.log(`✓ Created releases/${path.basename(crxPath)} (${(crxBuffer.length / 1024).toFixed(1)} KB)`);
