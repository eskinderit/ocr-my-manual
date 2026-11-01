// Local image-conditioned OCR probe. Prompts contain no expected brand/model name.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

async function main() {
  const model = process.argv[2] || 'qwen3vl4b';
  const port = process.argv[3] || '8003';
  const out = path.join(__dirname, 'outputs', 'logo_ocr', model);
  fs.mkdirSync(out, {recursive: true});
  const prompts = [
    'Transcribe all visible text in this image exactly as written. Preserve the original language. Output only the transcription, no explanation.',
    'Read the lettering in the image, including stylized lettering. Return only the visible text, with no commentary.'
  ];
  const inputs = [
    ['full_cover', 'imgs/vespa_manual_0.png'],
    ['isolated_logo', 'outputs/crops/vespa_manual_0/vespa_manual_0_3.png'],
    ['paragraph_control', 'outputs/crops/vespa_manual_1/vespa_manual_1_2.png']
  ];
  const records = [];
  for (const [id, relative] of inputs) {
    const bytes = fs.readFileSync(path.join(__dirname, relative));
    for (let p = 0; p < (id === 'paragraph_control' ? 1 : prompts.length); p++) {
      const prompt = model === 'glmocr' ? 'Text Recognition:' : prompts[p];
      const request = {
        model, messages: [{role: 'user', content: [
          {type:'image_url', image_url:{url:'data:image/png;base64,' + bytes.toString('base64')}},
          {type:'text', text:prompt}
        ]}], temperature:0, top_p:1, seed:42, max_tokens:2048, stream:false
      };
      const start = performance.now();
      const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(request)
      });
      const raw = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(raw));
      const record = {model, image:relative, image_sha256:crypto.createHash('sha256').update(bytes).digest('hex'),
        prompt, request_config:{temperature:0, top_p:1, seed:42, max_tokens:2048},
        elapsed_seconds:(performance.now()-start)/1000,
        text:raw.choices?.[0]?.message?.content, finish_reason:raw.choices?.[0]?.finish_reason, response:raw};
      fs.writeFileSync(path.join(out, `${id}-prompt${p+1}.json`), JSON.stringify(record,null,2)+'\n');
      console.log(id, 'prompt', p+1, record.elapsed_seconds.toFixed(3)+'s', JSON.stringify(record.text));
      records.push(record);
    }
  }
  fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(records,null,2)+'\n');
}
main().catch(e=>{console.error(e);process.exitCode=1;});