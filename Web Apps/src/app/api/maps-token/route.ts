import { NextResponse } from 'next/server';

export async function GET() {
  // Jika kamu menggunakan endpoint token dari referensi:
  // Kamu bisa fetch ke service token kamu di sini, lalu kembalikan ke client.
  // Tapi jika kamu punya Primary Key biasa, cukup kembalikan dari .env
  
  return NextResponse.json({ 
    token: process.env.AZURE_MAPS_SUBSCRIPTION_KEY,
    clientId: process.env.AZURE_MAPS_CLIENT_ID 
  });
}