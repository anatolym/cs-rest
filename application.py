# -*- coding: utf-8 -*-
"""
    CS - ColorSeason
    ~~~~~~
    ColorSeason application for image recognition (classification) by trained neural network model.
    :copyright: (c) 2016 by Anatoly Milkov <anatoly.milko@gmail.com>.
"""
import os
from datetime import datetime
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory, jsonify, abort
from werkzeug.utils import secure_filename
import json

from network import Network
import tools


# create our little application :)
application = Flask(__name__)
# Uploads config
UPLOAD_FOLDER = os.path.join(application.root_path, 'uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
IMAGE_MAX_SIZE = 16 * 1024 * 1024  # Image filesize limit.


# Load default config and override config from an environment variable
application.config.update(dict(
    DATABASE=os.path.join(application.root_path, 'cs-rest.sqlite3'),
    DEBUG=True,
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default',
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    # MAX_CONTENT_LENGTH=IMAGE_MAX_SIZE  # Doesn't work =/
))
application.config.from_envvar('FLASKR_SETTINGS', silent=True)

# Loading network.
net = Network()
net.load_model(application.root_path)


def connect_db():
    """Connects to the specific database."""
    rv = sqlite3.connect(application.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv


def init_db():
    """Initializes the database."""
    db = get_db()
    with application.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()


def get_db():
    """Opens a new database connection if there is none yet for the current application context."""
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db


def query_db(query, args=(), one=False):
    """Executes query and returns the result.
    """
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


@application.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

"""CLI actions."""


@application.cli.command('initdb')
def initdb_command():
    """Creates the database tables."""
    init_db()
    print('Initialized the database.')


@application.cli.command('run_comparison')
def run_comparison_command():
    filenames = [('/colorseason/Database/Prepared/train.txt', 'train'),
                 ('/colorseason/Database/Prepared/test.txt', 'test')]
    for filename in filenames:
        filelist = tools.get_filelist(filename[0])
        for image in filelist:
            # Skipping processed images.
            image_exists = query_db(
                'select comparison_id from image_comparison where filepath = ?', [image[0]], one=True)
            if image_exists:
                print('==> Image skipped (%s).' % image[0])
                continue

            image_filename = os.path.basename(image[0])
            result = net.test_image(image[0])
            status = 'true' if result['class_id'] == image[1] else 'false'
            time_processed = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            db = get_db()
            db.execute('insert into image_comparison ' +
                       '(phase, filename, filepath, origin_class, status, defined_class, defined_probability, defined_top, time_processed) ' +
                       'values (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                       [filename[1], image_filename, image[0], image[1], status,
                        result['class_id'], result['class_probability'],
                        json.dumps(result['top_inds']),
                        time_processed])
            db.commit()
            print('==> Image processed ("%s"), status: %s.' %
                  (image[0], status))
    print('Comparison is completed.')


def allowed_file(filename):
    filename = filename.lower()
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def ensure_upload_dir():
    if not os.path.isdir(application.config['UPLOAD_FOLDER']):
        os.makedirs(application.config['UPLOAD_FOLDER'])


def get_new_filename(filename):
    filename = filename.lower()
    name = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    ext = filename.rsplit('.', 1)[1]
    return name + '.' + ext


def purge_uploads():
    if os.path.isdir(application.config['UPLOAD_FOLDER']):
        for filename in os.listdir(application.config['UPLOAD_FOLDER']):
            filepath = os.path.join(
                application.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(e)


def get_image_list_in_uploads():
    image_list = []
    if os.path.isdir(application.config['UPLOAD_FOLDER']):
        for filename in os.listdir(application.config['UPLOAD_FOLDER']):
            filepath = os.path.join(
                application.config['UPLOAD_FOLDER'], filename)
            if allowed_file(filename) and os.path.isfile(filepath):
                image_list.append(filename)
    return image_list


@application.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('This filetype is not allowed')
            return redirect(request.url)
        if file:
            ensure_upload_dir()
            filename = get_new_filename(file.filename)
            filepath = os.path.join(
                application.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('classify', filename=filename))

    image_list = get_image_list_in_uploads()

    return render_template('index.html', image_list=image_list)


@application.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(application.config['UPLOAD_FOLDER'], filename)


@application.route("/classify/<filename>", methods=['GET'])
def classify(filename):
    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    image_url = None
    net_result = None
    if os.path.isfile(filepath):
        image_url = url_for('uploaded_file', filename=filename)

        # Running classification.
        net_result = net.test_image(filepath)
    return render_template('classify.html', image_url=image_url, net_result=net_result)


@application.route("/clear-uploads/", methods=['GET'])
def clear_uploads():
    purge_uploads()
    return redirect(url_for('index'))


@application.route("/test_api/", methods=['GET'])
def test_api():
    return render_template('test_api.html')


@application.route("/api/1.0/classify/", methods=['POST'])
def api_classify():
    # Check if the post request has the file part.
    if 'image' not in request.files:
        abort(400)
    file = request.files['image']
    # If user does not select file, browser also submit a empty part without
    # filename.
    if file.filename == '':
        abort(400)
    if not allowed_file(file.filename):
        abort(400)
    if not file:
        abort(400)

    ensure_upload_dir()
    filename = get_new_filename(file.filename)
    filepath = os.path.join(
        application.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Running classification.
    if not os.path.isfile(filepath):
        abort(400)

    image_url = url_for('uploaded_file', filename=filename)
    net_result = net.test_image(filepath)

    return jsonify(net_result)


@application.route("/comparison_results/", methods=['GET'])
def comparison_results():
    sql = "select * from image_comparison order by origin_class, phase, status, defined_probability DESC"
    image_list = query_db(sql)
    url_list = []
    for image in image_list:
        url_list.append(image['filepath'].lstrip('/'))
    total=len(image_list)
    train_count = query_db(
        'select count(comparison_id) from image_comparison where phase = "train"', one=True)
    train_count = train_count[0] if train_count else 0
    true_count = query_db(
        'select count(comparison_id) from image_comparison where status = "true"', one=True)
    true_count = true_count[0] if true_count else 0
    return render_template(
        'comparison_results.html',
        image_list=image_list,
        url_list=url_list,
        sql=sql,
        total=total,
        train_count=train_count,
        test_count=total-train_count,
        true_count=true_count,
        false_count=total-true_count)


@application.route('/images/', methods=['GET'])
def result_file():
    f = request.args.get('f', '')
    if ('colorseason/' not in f):
        abort(404)

    f = os.path.join('/', f)
    directory = os.path.dirname(f)
    filename = os.path.basename(f)
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    application.run('0.0.0.0', 5000, debug=True)
