// اضافه کردن این خط به بالاترین قسمت کد، مشکل SSL را به طور کامل نادیده می‌گیرد
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

require('dotenv').config();
const WebSocket = require('ws');
const http = require('http');
const https = require('https');
const { URL } = require('url');

// ... ادامه کدهای قبلی شما دقیقاً به همان شکل ...

// ==========================================
// تنظیمات پایه از فایل .env
// ==========================================
const WS_BASE_URL = process.env.WS_SERVER_URL || 'ws://127.0.0.1';
const WS_TOKEN = process.env.MICROSERVICE_WS_TOKEN;
const WS_SERVER_URL = `${WS_BASE_URL}/ws-microservice?token=${WS_TOKEN}`;

const LOCAL_SERVER_URL = process.env.LOCAL_SERVER_URL || 'http://127.0.0.1:100';
const RECONNECT_INTERVAL = 5000;
// ==========================================

let ws = null;
let isConnecting = false;

function connect() {
    if (isConnecting || (ws && ws.readyState === WebSocket.OPEN)) return;

    isConnecting = true;
    
    // چاپ بخشی از توکن برای اطمینان از اینکه توکن به درستی خوانده شده است
    const maskedToken = WS_TOKEN ? `${WS_TOKEN.substring(0, 4)}...${WS_TOKEN.slice(-4)}` : 'UNDEFINED';
    console.log(`\n[Socket] ---------------------------------------------`);
    console.log(`[Socket] Attempting to connect...`);
    console.log(`[Socket] Target URL: ${WS_BASE_URL}/ws-microservice`);
    console.log(`[Socket] Token provided: ${maskedToken}`);

    // استفاده از هدر x-auth-token در کنار Query String برای اطمینان بیشتر
    ws = new WebSocket(WS_SERVER_URL, {
        headers: {
            'x-auth-token': WS_TOKEN,
            rejectUnauthorized: false
        }
    });

    // رویداد اتصال موفق
    ws.on('open', () => {
        isConnecting = false;
        console.log('[Socket] ✅ Authenticated and connected successfully!');
    });

    // مهم‌ترین بخش برای دیباگ: رویداد خطای Handshake (مثل 401 یا 404)
    ws.on('unexpected-response', (request, response) => {
        isConnecting = false;
        console.error(`[Socket] ❌ Handshake failed! The server rejected the connection.`);
        console.error(`[Socket] HTTP Status: ${response.statusCode} ${response.statusMessage}`);
        
        // خواندن بادی خطای دریافتی از سرور برای لاگ دقیق‌تر
        response.on('data', (chunk) => {
            console.error(`[Socket] Server Response Body: ${chunk.toString()}`);
        });
    });

    // رویداد دریافت پیام
    ws.on('message', async (data) => {
        try {
            const requestData = JSON.parse(data.toString());
            await handleIncomingRequest(requestData);
        } catch (error) {
            console.error('[Socket] ⚠️ Invalid JSON payload received:', error.message);
        }
    });

    // رویداد قطع ارتباط
    ws.on('close', (code, reason) => {
        isConnecting = false;
        const reasonMsg = reason && reason.length > 0 ? reason.toString() : 'No explicit reason provided';
        console.log(`[Socket] 🔌 Connection closed. Code: ${code}, Reason: ${reasonMsg}`);
        console.log(`[Socket] Retrying in ${RECONNECT_INTERVAL / 1000} seconds...`);
        ws = null;
        setTimeout(connect, RECONNECT_INTERVAL);
    });

    // خطاهای سطح شبکه (مثل تایم‌اوت، در دسترس نبودن اینترنت یا قطع بودن سرور هدف)
    ws.on('error', (err) => {
        isConnecting = false;
        console.error('[Socket] ❌ Network/Socket Error:', err.message);
        if (err.code) console.error(`[Socket] Error Code: ${err.code}`);
    });
}

/**
 * پردازش درخواست و هدایت آن به لوکال سرور
 */
async function handleIncomingRequest(requestData) {
    const { requestId, method = 'GET', path, query = {}, body, headers = {} } = requestData;

    if (!requestId || !path) {
        console.error('[Proxy] Missing "requestId" or "path" in payload.');
        return;
    }

    const targetUrl = new URL(path, LOCAL_SERVER_URL);
    if (query && typeof query === 'object') {
        for (const [key, value] of Object.entries(query)) {
            targetUrl.searchParams.append(key, value);
        }
    }

    console.log(`[Proxy] Forwarding ${method} ${targetUrl.pathname}${targetUrl.search}`);

    delete headers['host'];
    delete headers['connection'];

    const options = {
        method: method,
        headers: headers
    };

    const reqModule = targetUrl.protocol === 'https:' ? https : http;

    const localReq = reqModule.request(targetUrl, options, (localRes) => {
        const chunks = [];

        localRes.on('data', (chunk) => chunks.push(chunk));

        localRes.on('end', () => {
            const responseBuffer = Buffer.concat(chunks);
            const contentType = localRes.headers['content-type'] || 'application/octet-stream';

            const isText = contentType.includes('text') || 
                           contentType.includes('json') || 
                           contentType.includes('xml');

            // ۲. ساخت ساختار دقیق JSON برای برگشت به سرور سوکت
            const responsePayload = {
                requestId: requestId,
                status: localRes.statusCode,
                contentType: contentType,
                // اضافه کردن هدرها به پاسخ ارسالی (این شیء شامل 'set-cookie' هم خواهد بود)
                headers: localRes.headers, 
                data: isText ? responseBuffer.toString('utf8') : responseBuffer.toString('base64'),
                isBase64: !isText
            };

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(responsePayload));
                console.log(`[Proxy] ✅ Replied to requestId: ${requestId} | Status: ${localRes.statusCode}`);
            }
        });
    });

    localReq.on('error', (err) => {
        console.error(`[Proxy] ❌ Local server error for ${path}:`, err.message);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                requestId: requestId,
                status: 502,
                contentType: 'application/json',
                data: JSON.stringify({ error: "Bad Gateway", message: "Microservice local server is unreachable." }),
                isBase64: false
            }));
        }
    });

    if (body && Object.keys(body).length > 0) {
        const bodyStr = typeof body === 'object' ? JSON.stringify(body) : body;
        localReq.write(bodyStr);
    }

    localReq.end();
}

connect();