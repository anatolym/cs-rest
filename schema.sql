drop table if exists image_comparison;
create table image_comparison (
  comparison_id integer primary key autoincrement,
  phase varchar(8) not null,
  filename varchar(512) not null,
  filepath varchar(512) not null,
  origin_class varchar(2) not null,
  status varchar(8) not null,
  defined_class varchar(2) not null,
  defined_probability decimal not null,
  defined_top text not null,
  time_processed datetime not null
);
