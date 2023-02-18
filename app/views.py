from app import app 
from app.tasks import inferance_model,queue_turn
from app.face_crop import image_croping
from flask import  request,jsonify,make_response ,render_template ,redirect, url_for
import os 
from zipfile import ZipFile
import secrets
import shutil
import datetime
import cv2
import base64
from app import r ,q

from rq.command import send_stop_job_command



def report_success(job, connection, result, *args, **kwargs):
    print("--------------------ssssssssssss------------------------")
    # print(job)
    print(connection)
    print(result)
    print("---------------------------------------------")

def report_failure(job, connection, type, value, traceback):
    print("--------------------ffffffffffffff----------------------")
    print(job)
    print(connection)
    print(type)
    print("---------------------------------------------")

if not os.path.isdir(app.config['USERS']):
    os.mkdir(app.config['USERS'])
if not os.path.isdir(app.config['UPLOAD']):
    os.mkdir(app.config['UPLOAD'])
HOME_PATH = os.path.join(os.getcwd(),'app')

JOBSIDS= []
@app.route("/")
def home():
    return "<h1>stable difiusion</h1>"

@app.route('/runwayml',methods=['POST'])
def yamlmodel():
    #args control ?
    args = request.args 

    if 'usr' in args :
        if len(args['usr']) > 1 :
            user_folder = os.path.join(app.config['USERS'],args['usr'] )
            if not os.path.isdir(user_folder): 
                os.mkdir(user_folder)
            try : 
                os.mkdir(os.path.join(user_folder,"models"))
                os.mkdir(os.path.join(user_folder,"genrated_images"))
                
            except:
                pass
        else:
            return make_response(jsonify({'message':False,'discription':'invalid username'}),400)
    else:
        return make_response(jsonify({'message':False,'discription':'no user'}),400)
    # ["face","half","close","full"]
    cut_type = "face"
    if 'cut-type' in args :
        if args['cut-type'] in ["face","half","close","full"] :
            cut_type = args['cut-type']

    instance_prompt = "secourses"
    if 'instance_prompt' in  args :
        if len(args['instance_prompt'] ) > 2:
            instance_prompt = args['instance_prompt']
    
    class_prompt = "portrait photo of a person"
    if 'class_prompt' in  args :
        if len(args['class_prompt'] ) > 2:
            class_prompt = args['class_prompt']

    learning_rate = 1e-6
    if 'learning_rate' in  args :
        if len(args['learning_rate'] ) > 2:
            try :
                learning_rate = float (args['learning_rate'])
            except :
                print("enter valid learningrate")

    lr_scheduler = "constant"
    if 'lr_scheduler' in  args :
        if len(args['lr_scheduler'] ) > 2:
            lr_scheduler = args['lr_scheduler']
    
    lr_warmup_steps = 208
    if 'lr_warmup_steps' in  args :
        try :
            lr_warmup_steps = int(args['lr_warmup_steps'])
        except :
            print("enter valid lr_warmup_steps")

    num_class = 312
    if 'num_class' in  args :
        try :
            num_class = int(args['num_class'])
        except :
            print("enter valid num_class")
    
    train_steps = 2080
    if 'train_steps' in  args :
        try :
            train_steps = int(args['train_steps'])
        except :
            print("enter valid train_steps")

    save_sample_prompt = "secourses"
    if 'save_sample_prompt' in  args :
        if len(args['save_sample_prompt'] ) > 2:
            save_sample_prompt = args['save_sample_prompt']
    #################inferance#######
    prompt = "face portrait of secourses, detailed face, insanely detailed, concept art, elegant, digital painting, looking into the camera"
    if 'prompt' in  args :
        if len(args['prompt'] ) > 2:
            prompt = args['prompt']

    negative_prompt = "bad anatomy, bw, black and white, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry bad anatomy, blurred"
    if 'negative_prompt' in  args :
        if len(args['negative_prompt'] ) > 2:
            negative_prompt = args['negative_prompt']

    num_samples = 5
    if 'num_samples' in  args :
        try :
            num_samples = int(args['num_samples'])
        except :
            print("enter valid num_samples")
    
    guidance_scale = 7.5
    if 'guidance_scale' in  args :
        try :
            guidance_scale = float(args['guidance_scale'])
        except :
            print("enter valid guidance_scale")
    
    num_inference_steps = 50
    if 'num_inference_steps' in  args :
        try :
            num_inference_steps = int(args['num_inference_steps'])
        except :
            print("enter valid num_inference_steps")

    height = 512
    if 'height' in  args :
        try :
            height = int(args['height'])
        except :
            print("enter valid height")

    width = 512
    if 'width' in  args :
        try :
            width = int(args['width'])
        except :
            print("enter valid width")
    # print(request.args)
    # print(lr_scheduler,learning_rate,lr_warmup_steps,num_class,train_steps,save_sample_prompt,instance_prompt,class_prompt,prompt 
    #             ,negative_prompt ,num_samples ,guidance_scale,num_inference_steps,height,width,)
    # return make_response(jsonify({'message':False,'discription':"error in the images"}),200)
    if request.method == "POST" :
        req = request.files

        if "zipFile" in req :
            # if True:SS
            if req["zipFile"].filename.split('.')[-1] =='zip' :

                folder_name = secrets.token_urlsafe(6)
                os.mkdir(os.path.join(app.config["UPLOAD"],folder_name))

                save_path = os.path.join(app.config["UPLOAD"],folder_name,req["zipFile"].filename) 
                req["zipFile"].save(save_path)
                
                with ZipFile (save_path,'r') as zip:
                    zip.extractall(path=os.path.join(app.config["UPLOAD"],folder_name))
                os.mkdir(os.path.join(app.config["UPLOAD"],folder_name +'input'))
                images_number =image_croping(os.path.join(app.config["UPLOAD"],folder_name),
                            os.path.join(app.config["UPLOAD"],folder_name+'input'),cut_on=cut_type)
                # images_number = 30
                # time.sleep(2)
                shutil.rmtree(os.path.join(app.config["UPLOAD"],folder_name), ignore_errors=True)
                model_path = secrets.token_urlsafe(6)+'_'+str(train_steps)
                output_dir =  os.path.join(user_folder,"models",model_path)
                weights_dir_path = os.path.join(output_dir,str(train_steps))
                genrate_save_path = datetime.datetime.strftime(datetime.datetime.now(),"%y-%m-%d&%H-%M-%S")+"_"+str(train_steps)+"_"+secrets.token_urlsafe(6)
                save_dir = os.path.join(user_folder,"genrated_images",genrate_save_path)
                os.mkdir(save_dir)
                
                arguments = {"images_path":os.path.join(app.config["UPLOAD"],folder_name+'input'),
                            "photo_path":os.path.join(app.config["STATIC"],"model_yaml","photos"),
                            "Model_name":"runwayml/stable-diffusion-v1-5",
                            "vae_name":"stabilityai/sd-vae-ft-mse",
                            "output_dir":output_dir,
                            "lr_scheduler":lr_scheduler,
                            "learning_rate":learning_rate,
                            "lr_warmup_steps":lr_warmup_steps,
                            "num_class":num_class,
                            "train_step":train_steps,
                            "save_sample_prompt":save_sample_prompt,
                            "instance_prompt":instance_prompt,
                            "class_prompt":class_prompt,
                            "weights_dir_path":weights_dir_path,
                            "save_dir":save_dir,
                            "prompt" :prompt ,"negative_prompt":negative_prompt ,
                            "num_samples":num_samples ,"guidance_scale":guidance_scale,
                            "num_inference_steps":num_inference_steps,
                            "height":height,"width":width,
                            "usr":args['usr'],"method":"train"}
                
                if images_number > 15 :
                    job = q.enqueue(queue_turn,**arguments,on_success=report_success, on_failure=report_failure)
                    # job = q.enqueue(sle,20,usr="ahmed")
                    JOBSIDS.append(job.id)
                    if len(JOBSIDS)>500 :
                        JOBSIDS [len(JOBSIDS)-500:]

                    return make_response(jsonify({'message':True,"discription":f"job enterd to the queue with total images {images_number}",
                    "weights_dir_path":model_path,"images_dir_path":genrate_save_path}))
                else :

                    return make_response(jsonify({'message':False,'discription':"error in the images"}),400)
            else :

                return make_response(jsonify({'message':False,'discription':'only files with zip extention'}),400)
        else :

            return make_response(jsonify({'message':False,'discription':'no image pass as "zipFile" in  body'}),400)
    else :
        return make_response(jsonify({'message':False,'discription':'Method Not Allowed'}),405)
#########################################inferance
@app.route('/genrate',methods=['POST'])
def genrate():
# args control ?
    args = request.args 
    
    if 'usr' in args :
        if len(args['usr']) > 1 :
            user_folder = os.path.join(app.config['USERS'],args['usr'] )
            if not os.path.isdir(user_folder): 
                os.mkdir(user_folder)
            try : 
                os.mkdir(os.path.join(user_folder,"genrated_images"))
                
            except:
                pass
        else:
            return make_response(jsonify({'message':False,'discription':'invalid username'}),400)
    else:
        return make_response(jsonify({'message':False,'discription':'no user'}),400)
    
    prompt = "face portrait of secourses, detailed face, insanely detailed, concept art, elegant, digital painting, looking into the camera"
    if 'prompt' in  args :
        if len(args['prompt'] ) > 2:
            prompt = args['prompt']

    negative_prompt = "bad anatomy, bw, black and white, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry bad anatomy, blurred"
    if 'negative_prompt' in  args :
        if len(args['negative_prompt'] ) > 2:
            negative_prompt = args['negative_prompt']

    num_samples = 5
    if 'num_samples' in  args :
        try :
            num_samples = int(args['num_samples'])
        except :
            print("enter valid num_samples")
    
    guidance_scale = 7.5
    if 'guidance_scale' in  args :
        try :
            guidance_scale = float(args['guidance_scale'])
        except :
            print("enter valid guidance_scale")
    
    num_inference_steps = 50
    if 'num_inference_steps' in  args :
        try :
            num_inference_steps = int(args['num_inference_steps'])
        except :
            print("enter valid num_inference_steps")

    height = 512
    if 'height' in  args :
        try :
            height = int(args['height'])
        except :
            print("enter valid height")

    width = 512
    if 'width' in  args :
        try :
            width = int(args['width'])
        except :
            print("enter valid width")
    train_steps = 800
    model_path= "default"
    weights_dir_path = os.path.join(app.config["STATIC"],"model_yaml","diffusers","models--runwayml--stable-diffusion-v1-5","snapshots","ded79e214aa69e42c24d3f5ac14b76d568679cc2")
    if 'weights_dir_path' in  args :
        try :
            weihgts_path =os.path.join(user_folder,'models',args['weights_dir_path'].split('_')[0],args['weights_dir_path'].split('_')[1]) 
            train_steps= args['weights_dir_path'].split('_')[1]
            model_path= args['weights_dir_path']
            if os.isdir(weihgts_path):
                print('found_path')
                weights_dir_path = weihgts_path
                print(weights_dir_path)
        except :
            print("enter valid width")
# end args  
    if request.method == "POST" :
        genrate_save_path=datetime.datetime.strftime(datetime.datetime.now(),"%y-%m-%d&%H-%M-%S")+"_"+str(train_steps)+"_"+secrets.token_urlsafe(6)
        save_dir = os.path.join(user_folder,"genrated_images",genrate_save_path)
        try:
            os.mkdir(save_dir)
        except:
            print("could'nt make the dir")
        
        arguments = {
                    "weights_dir_path":weights_dir_path,
                    "save_dir":save_dir,
                    "prompt" :prompt ,"negative_prompt":negative_prompt ,
                    "num_samples":num_samples ,"guidance_scale":guidance_scale,
                    "num_inference_steps":num_inference_steps,
                    "height":height,"width":width,
                    "usr":args['usr'],"method":'genrate'}
        
        job = q.enqueue(inferance_model,**arguments,on_success=report_success, on_failure=report_failure)
            # job = q.enqueue(sle,20,usr="ahmed")
        JOBSIDS.append(job.id)
        if len(JOBSIDS)>500 :
            JOBSIDS [len(JOBSIDS)-500:]

        return make_response(jsonify({'message':True,"discription":f"job enterd to the queue with total images ",
        "weights_dir_path":model_path,"images_dir_path":genrate_save_path}))
        
    else :
        return make_response(jsonify({'message':False,'discription':'Method Not Allowed'}),405)

@app.route('/listmodels',methods=["GET"])
def listmodels():
    # args control ?
    args = request.args 
    
    if 'usr' in args :
        if args['usr'] in os.listdir(app.config['USERS']):

            user_folder = os.path.join(app.config['USERS'],args['usr'],'models' )
            if os.path.isdir(user_folder):
                models = os.listdir(user_folder)
                return make_response(jsonify({'message':True,'models':models}),200)
            else:
                return make_response(jsonify({'message':False,'discription':'user has not build any model yet'}),400)    
                
        else:
            return make_response(jsonify({'message':False,'discription':'user dose not exist'}),400)
    else:
        return make_response(jsonify({'message':False,'discription':'no user'}),400)

@app.route('/listimages',methods=["GET"])
def listimages():
    # args control ?
    # 23-01-31&02-45-25_200_gP0VvJCf
    # 23-01-31&02-45-25_200_gP0VvJCf
    args = request.args 
    
    if 'usr' in args :
        if args['usr'] in os.listdir(app.config['USERS']):

            user_folder = os.path.join(app.config['USERS'],args['usr'],'genrated_images' )
            if os.path.isdir(user_folder):
                images_folders = os.listdir(user_folder)
                images={}
                for fo in images_folders:
                    images_list=[]
                    for im in os.listdir(os.path.join(user_folder,fo)):
                        try:
                            _, buffer = cv2.imencode('bb.jpg', cv2.imread(os.path.join(os.path.join(user_folder,fo,im))))
                            images_list.append(str(base64.b64encode(buffer))[2:-1])
                        except :
                            print("faild")
                    images.update({fo:images_list})
                return make_response(jsonify({'message':True,'images':images}),200)
            else:
                return make_response(jsonify({'message':False,'discription':'user has not genrated images yet'}),400)    
                
        else:
            return make_response(jsonify({'message':False,'discription':'user dose not exist'}),400)
    else:
        return make_response(jsonify({'message':False,'discription':'no user'}),400)
@app.route("/jobs",methods=['GET'])
def wor():
    jobs = []   
    for jj in JOBSIDS:#q.job_ids :
        job = q.fetch_job(jj)
        if job is not None:
            jl = [jj, job.get_status(),job.enqueued_at.strftime('%m/%d/%Y %H:%M:%S'),]
            if job.started_at is not None:
                jl.append(job.started_at.strftime('%m/%d/%Y %H:%M:%S'))
            else :
                jl.append(None)
            if  job.ended_at is not None :
                jl.append(job.ended_at.strftime('%m/%d/%Y %H:%M:%S'))
            else :
                jl.append(None)
            if  job.kwargs['usr'] is not None :
                jl.append(job.kwargs['usr'])
            else :
                jl.append(None)
            if  job.kwargs['method'] is not None :
                jl.append(job.kwargs['method'])
            else :
                jl.append(None)

            jl.append(job.result)
            jobs.append(jl)
        
    return render_template("public/log_table.html",log_table=jobs)
import time
@app.route('/cancel',methods=['GET'])
def cancel ():
    if 'id' in request.args:
        
        if request.args['id'] in q.job_ids  :
            
            try :
            
                job = q.fetch_job(request.args['id'])
                job.cancel()
                
            except:
                print("faild to cancel the job")
        else :
            try:
                send_stop_job_command(r, request.args['id'])
                time.sleep(2)
                return redirect("/jobs")
            except:
                print("job not found")
    else :
        print("no id sent")

    return redirect("/jobs")


