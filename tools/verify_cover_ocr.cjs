const fs = require('node:fs');
const crypto = require('node:crypto');
async function main() {
  const image = 'imgs/vespa_manual_0.png';
  const bytes = fs.readFileSync(image);
  const prompt = 'Transcribe all visible lettering, including single letters inside emblems, stylized words, dates and numbers. Preserve the original language and spelling. Output only visible text. Do not describe the image.';
  const payload = {model:'qwen3vl4b', messages:[{role:'user',content:[{type:'image_url',image_url:{url:'data:image/png;base64,'+bytes.toString('base64')}},{type:'text',text:prompt}]}],temperature:0,top_p:1,seed:42,max_tokens:2048};
  const start = performance.now();
  const response = await fetch('http://127.0.0.1:8003/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const raw = await response.json();
  if(!response.ok) throw Error(JSON.stringify(raw));
  const result = {model:'qwen3vl4b',image,image_sha256:crypto.createHash('sha256').update(bytes).digest('hex'),prompt,elapsed_seconds:(performance.now()-start)/1000,text:raw.choices[0].message.content,finish_reason:raw.choices[0].finish_reason,response:raw};
  fs.writeFileSync('outputs/logo_ocr/qwen3vl4b/full-cover-emblem-check.json',JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
