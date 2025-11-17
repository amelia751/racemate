import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const telemetry = await request.json();
    
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8005';
    
    // Send telemetry to backend's realtime prediction endpoint
    const response = await fetch(`${backendUrl}/realtime/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ telemetry }),
    });
    
    if (!response.ok) {
      const error = await response.text();
      console.error('[Telemetry API] Backend error:', error);
      return NextResponse.json(
        { error: 'Backend processing failed', details: error },
        { status: response.status }
      );
    }
    
    const result = await response.json();
    console.log('[Telemetry API] Backend response:', result);
    
    return NextResponse.json({
      success: true,
      events: result.events || [],
      recommendations: result.recommendations || null,
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('[Telemetry API] Error:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: error.message },
      { status: 500 }
    );
  }
}

