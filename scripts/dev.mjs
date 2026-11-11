import {spawn} from 'node:child_process';
const api=spawn(process.env.PYTHON||'python3',['-m','uvicorn','afterlight.api:app','--host','127.0.0.1','--port','8000'],{stdio:'inherit',env:{...process.env,PYTHONPATH:'backend'}});
const ui=spawn('npm',['--prefix','frontend','run','dev','--',...process.argv.slice(2)],{stdio:'inherit'});
function stop(){api.kill();ui.kill()}
process.on('SIGINT',stop);process.on('SIGTERM',stop);
api.on('exit',code=>{if(code){stop();process.exitCode=code}});ui.on('exit',code=>{stop();process.exitCode=code??0});
